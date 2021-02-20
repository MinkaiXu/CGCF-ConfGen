import os
import time
import argparse
import torch
import pickle
from torch_geometric.data import DataLoader
from tqdm.auto import tqdm
from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMolecule

from models.edgecnf import *
from utils.dataset import *
from utils.chem import *
from utils.misc import *
from utils.transforms import *

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, default='./pretrained/ckpt_qm9.pt')
parser.add_argument('--dataset', type=str, default='./data/qm9/test.pkl')
parser.add_argument('--out', type=str, default='./generated.pkl')
parser.add_argument('--num_samples', type=int, default=-2, help=
                    'Fixed number of confs for each molecule if `num_samples` > 0.' +
                    '(-num_samples) times as many as confs in the test-set for each molecule if `num_samples` < 0'
                    )
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--batch_size', type=int, default=1000)
parser.add_argument('--emb_step_size', type=float, default=3.0) # 3.0 for QM9, 5.0 for ISO17
parser.add_argument('--emb_num_steps', type=int, default=1000)
parser.add_argument('--mmff', action='store_true', default=False)
args = parser.parse_args()

# Logging
logger = get_logger('gen', log_dir=None)
logger.info(args)

# Model
logger.info('Loading EdgeCNF...')
ckpt = torch.load(args.ckpt)
model_cnf = EdgeCNF(ckpt['args']).to(args.device)
if ckpt['args'].spectral_norm:
    add_spectral_norm(model_cnf)
model_cnf.load_state_dict(ckpt['state_dict'])

# Test Dataset
logger.info('Loading test-set: %s' % args.dataset)
tf = get_standard_transforms(ckpt['args'].aux_edge_order)
test_dset = MoleculeDataset(args.dataset, transform=tf)
grouped = split_dataset_by_smiles(test_dset)
loader = DataLoader(VirtualDataset(grouped, args.num_samples), batch_size=args.batch_size, shuffle=False)

# Output buffer
gen_rdmols = []

# DistGeom Embedder
embedder = Embed3D(step_size=args.emb_step_size, num_steps=args.emb_num_steps)

# Generate
all_data_list = []
for batch in tqdm(loader, 'Generate'):
    batch = batch.to(args.device)
    pos_s = simple_generate_batch(model_cnf, batch, num_samples=1, embedder=embedder)[0]  # (1, BN, 3)
    batch.pos = pos_s[0]
    batch.to('cpu')
    batch_list = batch.to_data_list()
    all_data_list += batch_list

grouped_data = split_dataset_by_smiles(all_data_list)
for smiles in grouped_data:
    for data in grouped_data[smiles]:
        rdmol = data['rdmol']
        rdmol = set_rdmol_positions_(rdmol, data.pos.cpu())
        gen_rdmols.append(rdmol)


# Optimize using MMFF
opt_rdmols = []
if args.mmff:
    for mol in tqdm(gen_rdmols, desc='MMFF Optimize'):
        opt_mol = deepcopy(mol)
        MMFFOptimizeMolecule(opt_mol)
        opt_rdmols.append(opt_mol)
    gen_rdmols = opt_rdmols

# Save
logger.info('Saving to: %s' % args.out)
with open(args.out, 'wb') as f:
    pickle.dump(gen_rdmols, f)
