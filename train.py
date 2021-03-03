import os
import argparse
import torch
import torch.utils.tensorboard
from torch_geometric.data import Batch, DataLoader
from tqdm.auto import tqdm

from models.edgecnf import *
from models.cnf_edge import NONLINEARITIES, LAYERS, SOLVERS
from utils.dataset import *
from utils.transforms import *
from utils.misc import *

# Arguments
parser = argparse.ArgumentParser()
# BEGIN
# Model arguments
parser.add_argument('--activation', type=str, default='softplus')
parser.add_argument('--hidden_dim', type=int, default=64)
parser.add_argument("--num_blocks", type=int, default=1,
                    help='Number of stacked CNFs.')
parser.add_argument("--layer_type", type=str, default="concatsquash", choices=LAYERS)
parser.add_argument('--time_length', type=float, default=0.5)
parser.add_argument('--train_T', type=eval, default=True, choices=[True, False])
parser.add_argument('--use_adjoint', type=eval, default=True, choices=[True, False])
parser.add_argument('--solver', type=str, default='dopri5', choices=SOLVERS)
parser.add_argument('--atol', type=float, default=1e-5)
parser.add_argument('--rtol', type=float, default=1e-5)
parser.add_argument('--batch_norm', type=eval, default=True, choices=[True, False])
parser.add_argument('--sync_bn', type=eval, default=False, choices=[True, False])
parser.add_argument('--bn_lag', type=float, default=0)
parser.add_argument('--spectral_norm', type=eval, default=True, choices=[True, False])
parser.add_argument('--train_noise_std', type=float, default=0.1)

# Datasets and loaders
parser.add_argument('--aux_edge_order', type=int, default=3)
parser.add_argument('--train_dataset', type=str, default='./data/qm9/train.pkl')
parser.add_argument('--val_dataset', type=str, default='./data/qm9/val.pkl')
parser.add_argument('--train_batch_size', type=int, default=128)
parser.add_argument('--val_batch_size', type=int, default=256)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--max_val_batch', type=int, default=5)

# Optimizer and scheduler
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--sched_factor', type=float, default=0.5)
parser.add_argument('--sched_patience', type=int, default=3,
                    help='Patience steps = sched_patience * val_freq')
parser.add_argument('--sched_min_lr', type=int, default=1e-5)
parser.add_argument('--beta1', type=float, default=0.95)
parser.add_argument('--beta2', type=float, default=0.999)

# Training
parser.add_argument('--seed', type=int, default=2020)
parser.add_argument('--logging', type=eval, default=True, choices=[True, False])
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--max_iters', type=int, default=50*1000, 
                    help='Max iterations for MLE pre-training of CNF')
parser.add_argument('--val_freq', type=int, default=300)
parser.add_argument('--inspect_freq', type=int, default=50)
parser.add_argument('--tag', type=str, default='')
parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--log_root', type=str, default='./logs')
# END
args = parser.parse_args()
seed_all(args.seed)

# Logging
if args.logging:
    log_dir = get_new_log_dir(root=args.log_root, prefix='ECNF', tag=args.tag)
    logger = get_logger('train', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    ckpt_mgr = CheckpointManager(log_dir)
    log_hyperparams(writer, args)
else:
    logger = get_logger('train', None)
    writer = BlackHole()
    ckpt_mgr = BlackHole()
logger.info(args)

# Datasets and loaders
logger.info('Loading dataset...')
tf = get_standard_transforms(order=args.aux_edge_order)
train_dset = MoleculeDataset(args.train_dataset, transform=tf)
val_dset = MoleculeDataset(args.val_dataset, transform=tf)
train_iterator = get_data_iterator(DataLoader(train_dset, batch_size=args.train_batch_size, shuffle=True, drop_last=True))
val_loader = DataLoader(val_dset, batch_size=args.val_batch_size, shuffle=False, drop_last=True)
logger.info('TrainSet %d | ValSet %d' % (len(train_dset), len(val_dset)))

# Model
logger.info('Building model...')
if args.resume is None:
    model = EdgeCNF(args).to(args.device)
    if args.spectral_norm:
        add_spectral_norm(model, logger=logger)
else:
    logger.info('Resuming from %s' % args.resume)
    ckpt_resume = CheckpointManager(args.resume, logger=logger).load_latest()
    model = EdgeCNF(ckpt_resume['args']).to(args.device)
    if ckpt_resume['args'].spectral_norm:
        add_spectral_norm(model, logger=logger)
    model.load_state_dict(ckpt_resume['state_dict'])
logger.info(repr(model))

# Optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(), 
    lr=args.lr, 
    weight_decay=args.weight_decay,
    betas=(args.beta1, args.beta2)
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
    factor=args.sched_factor,
    patience=args.sched_patience,
    min_lr=args.sched_min_lr
)

# Train and validation
def train(it):
    model.train()
    optimizer.zero_grad()
    if args.spectral_norm:
        spectral_norm_power_iteration(model, n_power_iterations=1)
    batch = next(train_iterator).to(args.device)
    noise = torch.randn_like(batch.edge_length) * args.train_noise_std
    loss = model.get_loss(batch, batch.edge_length + noise)
    nfe_forward = count_nfe(model)

    loss.backward()
    optimizer.step()

    nfe_total = count_nfe(model)
    nfe_backward = nfe_total - nfe_forward
    
    logger.info('[Train] Iter %04d | Loss %.6f | NFE_Forward %d | NFE_Backward %d ' % (it, loss.item(), nfe_forward, nfe_backward))
    writer.add_scalar('train/loss', loss, it)
    writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
    writer.add_scalar('train/nfe_forward', nfe_forward, it)
    writer.add_scalar('train/nfe_backward', nfe_backward, it)
    writer.flush()

def validate(it):
    with torch.no_grad():
        sum_loss = 0
        sum_n = 0
        model.eval()
        for i, batch in enumerate(tqdm(val_loader, desc='Validating')):
            if i >= args.max_val_batch:
                break
            batch = batch.to(args.device)
            log_pd = model.get_log_prob(batch, batch.edge_length)
            sum_loss += -log_pd.sum().item()
            sum_n += log_pd.size(0)
        avg_loss = sum_loss / sum_n

        scheduler.step(avg_loss)

        logger.info('[Validate] Iter %04d | Loss %.6f ' % (it, avg_loss))
        writer.add_scalar('val/loss', avg_loss, it)
        writer.flush()
    return avg_loss

def inspect(it):
    logger.info('[Inspect] Sampling edge lengths...')
    with torch.no_grad():
        molecule = Batch.from_data_list([val_dset[0]]).to(args.device)
        model.eval()
        samples, _ = model.sample(molecule, num_samples=500)       # (E, num_samples)
        for i, edge_name in enumerate(molecule.edge_name[0]):   # Only one molecule
            if edge_name == '':
                continue
            mean = samples[i].mean().item()
            std = samples[i].std().item()
            name_seg = edge_name.split('_')
            logger.info('[Inspect] (%d) %s %s-%s | Dist %.6f | Mean %.6f | Std %.6f' % (
                i,
                name_seg[0],
                name_seg[1],
                name_seg[2],
                molecule.edge_length[i].item(),
                mean,
                std,
            ))
            writer.add_histogram('length/' + edge_name, samples[i], it)
        writer.flush()

# Main loop
logger.info('Start training...')
try:
    if args.resume is not None:
        start_it = ckpt_resume['iteration'] + 1
    else:
        start_it = 1
    for it in range(start_it, args.max_iters+1):
        train(it)
        if it % args.val_freq == 0 or it == args.max_iters:
            avg_val_loss = validate(it)
            ckpt_mgr.save(model, args, avg_val_loss, it)
        if it % args.inspect_freq == 0:
            inspect(it)

except KeyboardInterrupt:
    logger.info('Terminating...')
