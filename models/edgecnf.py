import torch

from .common import *
from .cnf_edge import CNF, ODEfunc, ODEgnn, MovingBatchNorm1d, SequentialFlow, count_nfe, add_spectral_norm, spectral_norm_power_iteration
from .distgeom import *


def build_flow(args, hidden_dim, num_blocks):
    def build_cnf():
        diffeq = ODEgnn(
            hidden_dim=hidden_dim,
        )
        odefunc = ODEfunc(
            diffeq=diffeq,
        )
        cnf = CNF(
            odefunc=odefunc,
            T=args.time_length,
            train_T=args.train_T,
            solver=args.solver,
            use_adjoint=args.use_adjoint,
            atol=args.atol,
            rtol=args.rtol,
        )
        return cnf

    chain = [build_cnf() for _ in range(num_blocks)]
    if args.batch_norm:
        bn_layers = [MovingBatchNorm1d(1, bn_lag=args.bn_lag, sync=args.sync_bn)
                     for _ in range(num_blocks)]
        bn_chain = [MovingBatchNorm1d(1, bn_lag=args.bn_lag, sync=args.sync_bn)]
        for a, b in zip(chain, bn_layers):
            bn_chain.append(a)
            bn_chain.append(b)
        chain = bn_chain
    model = SequentialFlow(chain)
    return model


class EdgeCNF(torch.nn.Module):

    def __init__(self, args):
        super().__init__()
        self.node_emb = torch.nn.Embedding(100, args.hidden_dim)
        self.edge_emb = torch.nn.Embedding(100, args.hidden_dim)
        self.flow = build_flow(
            args,
            hidden_dim=args.hidden_dim,
            num_blocks=args.num_blocks,
        )

    def get_d(self, data, z):
        node_attr = self.node_emb(data.node_type)
        edge_attr = self.edge_emb(data.edge_type)
        d = self.flow(
            z,
            node_attr = node_attr,
            edge_attr = edge_attr,
            edge_index = data.edge_index,
            reverse = True,
        )
        return d

    def get_z(self, data, d):
        node_attr = self.node_emb(data.node_type)
        edge_attr = self.edge_emb(data.edge_type)
        z = self.flow(
            d,
            node_attr = node_attr,
            edge_attr = edge_attr,
            edge_index = data.edge_index,
            reverse = False
        )
        return z

    def get_log_prob(self, data, d):
        E = d.size(0)
        z, delta_logpz = self.flow(
            x = d,
            node_attr = self.node_emb(data.node_type),
            edge_attr = self.edge_emb(data.edge_type),
            edge_index = data.edge_index,
            logpx=torch.zeros(E, 1).to(d)
        )
        log_pz = standard_normal_logprob(z).view(E, -1).sum(1, keepdim=True)
        log_pd = log_pz - delta_logpz
        return log_pd

    def get_normalize_traj(self, data, d, integration_times):
        E = d.size(0)
        z_traj, _ = self.flow(
            x = d,
            node_attr = self.node_emb(data.node_type),
            edge_attr = self.edge_emb(data.edge_type),
            edge_index = data.edge_index,
            logpx=torch.zeros(E, 1).to(d),
            integration_times=integration_times,
        )
        return z_traj


    def get_loss(self, data, d):
        log_pd = self.get_log_prob(data, d)
        loss = - log_pd.mean()
        return loss

    def sample(self, data, num_samples):
        E = data.edge_index.size(1)
        node_attr = self.node_emb(data.node_type)
        edge_attr = self.edge_emb(data.edge_type)
        z = torch.randn(num_samples*E, 1).to(edge_attr)

        edge_index_rep = []
        for i in range(num_samples):
            edge_index_rep.append(data.edge_index + data.num_nodes * i)
        edge_index_rep = torch.cat(edge_index_rep, dim=1)

        samples = self.flow(
            z,
            node_attr = node_attr.repeat(num_samples, 1),
            edge_attr = edge_attr.repeat(num_samples, 1),
            edge_index = edge_index_rep,
            reverse = True
        )

        samples = samples.reshape(-1, E).t()    # (E, num_samples)
        return samples, z


def simple_generate_batch(model:EdgeCNF, data, num_samples, embedder=Embed3D(), dg_init_pos=None):
    """
    Generate conformations in batch using only EdgeCNF.
    """
    with torch.no_grad():
        d, _ = model.sample(data, num_samples)  # (E, num_samples)
        d = d.t()

    edge_indices = []
    for i in range(num_samples):
        edge_indices.append(data.edge_index + i * data.num_nodes)
    edge_indices = torch.cat(edge_indices, dim=1)

    if dg_init_pos is None:
        dg_init_pos = torch.randn(num_samples*data.num_nodes, 3).to(data.pos)
    else:
        dg_init_pos = dg_init_pos.repeat(num_samples, 1)

    pos, _ = embedder(
        d.view(-1), 
        edge_indices,
        dg_init_pos,
        data.edge_order.repeat(num_samples),
    )   # (num_samples*N, 3)

    d_new = torch.norm(pos[edge_indices[0]] - pos[edge_indices[1]], dim=1).reshape(num_samples, -1)
    d_new = d_new.view(num_samples, -1)
    pos = pos.view(num_samples, -1, 3)

    return pos, d_new


