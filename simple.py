import argparse
import itertools
import time

import torch
from e3nn import o3
from e3nn.math import soft_one_hot_linspace
from e3nn.nn.models.v2103.gate_points_message_passing import MessagePassing
from torch_cluster import radius_graph
from torch_geometric.data import DataLoader
from torch_geometric.datasets import QM9
from torch_scatter import scatter

import wandb


class Network(torch.nn.Module):
    def __init__(
        self,
        muls,
        sh_lmax,
        num_layers=3,
        max_radius=10.0,
        num_basis=50,
        fc_neurons=[128, 128],
        num_neighbors=20,
        num_nodes=20,
        atomref=None,
    ) -> None:
        super().__init__()

        self.sh_lmax = sh_lmax
        self.max_radius = max_radius
        self.num_basis = num_basis
        self.num_nodes = num_nodes

        self.register_buffer('atomref', atomref)

        irreps_node_hidden = o3.Irreps([
            (mul, (l, p))
            for l, mul in enumerate(muls)
            for p in [-1, 1]
        ])

        self.mp = MessagePassing(
            irreps_node_input="0e",
            irreps_node_hidden=irreps_node_hidden,
            irreps_node_output="0e + 0o",
            irreps_node_attr="5x0e",
            irreps_edge_attr=o3.Irreps.spherical_harmonics(sh_lmax),
            layers=num_layers,
            fc_neurons=[self.num_basis] + fc_neurons,
            num_neighbors=num_neighbors,
        )

    def forward(self, data) -> torch.Tensor:
        node_atom = data['z']
        node_pos = data['pos']
        batch = data['batch']

        # The graph
        edge_src, edge_dst = radius_graph(
            node_pos,
            r=self.max_radius,
            batch=batch,
            max_num_neighbors=1000
        )

        # Edge attributes
        edge_vec = node_pos[edge_src] - node_pos[edge_dst]
        edge_sh = o3.spherical_harmonics(
            l=range(self.sh_lmax + 1),
            x=edge_vec,
            normalize=True,
            normalization='component'
        )

        # Edge length embedding
        edge_length = edge_vec.norm(dim=1)
        edge_length_embedding = soft_one_hot_linspace(
            edge_length,
            0.0,
            self.max_radius,
            self.num_basis,
            basis='smooth_finite',
            cutoff=True,
        ).mul(self.num_basis**0.5)

        node_input = node_pos.new_ones(node_pos.shape[0], 1)

        node_attr = node_atom.new_tensor([-1, 0, -1, -1, -1, -1, 1, 2, 3, 4])[node_atom]
        node_attr = torch.nn.functional.one_hot(node_attr, 5).mul(5**0.5)

        node_outputs = self.mp(
            node_features=node_input,
            node_attr=node_attr,
            edge_src=edge_src,
            edge_dst=edge_dst,
            edge_attr=edge_sh,
            edge_scalars=edge_length_embedding
        )

        node_outputs = node_outputs[:, 0] + node_outputs[:, 1].pow(2).mul(0.5)
        node_outputs = node_outputs.view(-1, 1)

        node_outputs = node_outputs.div(self.num_nodes**0.5)

        if self.atomref is not None:
            node_outputs = node_outputs + self.atomref[node_atom]
        # for target=7, MAE of 75eV

        outputs = scatter(node_outputs, batch, dim=0)

        return outputs


def execute(config):
    device = torch.device(config['device'])
    torch.manual_seed(config['seed'])

    # Report meV instead of eV.
    units = 1000 if config['target'] in [2, 3, 4, 6, 7, 8, 9, 10] else 1

    dataset = QM9(config['data_path'])
    train_dataset, val_dataset = dataset[:50000], dataset[50000:70000]

    model = Network(
        muls=(config['mul0'], config['mul1'], config['mul2']),
        sh_lmax=config['shlmax'],
        num_layers=config['num_layers'],
        max_radius=config['max_radius'],
        num_basis=config['num_basis'],
        fc_neurons=[config['radial_num_neurons']] * config['radial_num_layers'],
        num_neighbors=20.0,
        num_nodes=20.0,
        atomref=dataset.atomref(config['target']),
    )
    model = model.to(device)

    wandb.watch(model)

    optim = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=25, factor=0.5, verbose=True)

    runtime = time.perf_counter()
    runtime_print = time.perf_counter()

    for epoch in itertools.count():

        errs = []
        loader = DataLoader(train_dataset, batch_size=config['bs'], shuffle=True)

        for step, data in enumerate(loader):
            data = data.to(device)

            pred = model(data)
            err = pred.view(-1) - data.y[:, config['target']]

            optim.zero_grad()
            err.pow(2).mean().backward()
            optim.step()

            errs += [err.cpu().detach()]

            if time.perf_counter() - runtime_print > 15:
                runtime_print = time.perf_counter()
                w = time.perf_counter() - runtime
                e = epoch + (step + 1) / len(loader)
                print((
                    f'[{e:.1f}] ['
                    f'runtime={w / 3600:.2f}h '
                    f'runtime/epoch={w / e:.0f}s '
                    f'runtime/step={1e3 * w / e / len(loader):.0f}ms '
                    f'step={step}/{len(loader)} '
                    f'mae={units * torch.cat(errs)[-200:].abs().mean():.5f} '
                ), flush=True)

        train_err = torch.cat(errs)

        errs = []
        loader = DataLoader(val_dataset, batch_size=256)
        for data in loader:
            data = data.to(device)
            with torch.no_grad():
                pred = model(data)

            err = pred.view(-1) - data.y[:, config['target']]
            errs += [err.cpu().detach()]
        val_err = torch.cat(errs)

        lrs = [
            x['lr']
            for x in optim.param_groups
        ]
        status = {
            'epoch': epoch,
            '_runtime': time.perf_counter() - runtime,
            'train': {
                'mae': {
                    'mean': units * train_err.abs().mean().item(),
                    'std': units * train_err.abs().std().item(),
                },
                'mse': {
                    'mean': units * train_err.pow(2).mean().item(),
                    'std': units * train_err.pow(2).std().item(),
                }
            },
            'val': {
                'mae': {
                    'mean': units * val_err.abs().mean().item(),
                    'std': units * val_err.abs().std().item(),
                },
                'mse': {
                    'mean': units * val_err.pow(2).mean().item(),
                    'std': units * val_err.pow(2).std().item(),
                }
            },
            'lrs': lrs,
        }
        wandb.log(status)

        print((
            f'[{epoch}] Target: {config["target"]:02d}, '
            f'MAE TRAIN: {units * train_err.abs().mean():.5f} ± {units * train_err.abs().std():.5f}, '
            f'MAE VAL: {units * val_err.abs().mean():.5f} ± {units * val_err.abs().std():.5f}'
        ), flush=True)

        scheduler.step(val_err.pow(2).mean())

        if status['_runtime'] > config['max_runtime']:
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mul0", type=int, default=256)
    parser.add_argument("--mul1", type=int, default=16)
    parser.add_argument("--mul2", type=int, default=0)
    parser.add_argument("--shlmax", type=int, default=1)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--max_radius", type=float, default=10.0)
    parser.add_argument("--num_basis", type=int, default=50)
    parser.add_argument("--radial_num_neurons", type=int, default=128)
    parser.add_argument("--radial_num_layers", type=int, default=2)

    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--bs", type=int, default=50)

    parser.add_argument("--target", type=int, default=7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_runtime", type=int, default=(3 * 24 - 1) * 3600)
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--data_path", type=str, default='QM9')

    args = parser.parse_args()

    wandb.login()
    wandb.init(project=f"QM9 #{args.target}", config=args.__dict__)
    config = dict(wandb.config)
    print(config)
    execute(config)


if __name__ == "__main__":
    main()
