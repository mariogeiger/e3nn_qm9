import argparse
import datetime
import itertools
import pickle
import subprocess
import time

import torch
import wandb

from torch_geometric.data import DataLoader
from torch_geometric.datasets import QM9
from torch_geometric.nn import SchNet

from model import Network


def execute(config):
    path = 'QM9'
    dataset = QM9(path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(config['seed'])

    # Report meV instead of eV.
    units = 1000 if config['target'] in [2, 3, 4, 6, 7, 8, 9, 10] else 1

    _, datasets = SchNet.from_qm9_pretrained(path, dataset, config['target'])
    train_dataset, val_dataset, _test_dataset = datasets

    model = Network(
        muls=(config['mul0'], config['mul1'], config['mul2']),
        lmax=config['lmax'],
        num_layers=config['num_layers'],
        rad_gaussians=config['rad_gaussians'],
        rad_hs=(config['rad_h'],) * config['rad_layers'],
        mean=config['mean'], std=config['std'],
        atomref=dataset.atomref(config['target']),
    )
    model = model.to(device)

    wandb.watch(model)

    # modules = [model.embedding, model.radial] + list(model.layers) + [model.atomref]
    # lrs = [0.1, 0.01] + [1] * len(model.layers) + [0.1]
    # param_groups = []
    # for lr, module in zip(lrs, modules):
    #     jac = []
    #     for data in DataLoader(train_dataset[:20]):
    #         data = data.to(device)
    #         jac += [torch.autograd.grad(model(data.z, data.pos), module.parameters())[0].flatten()]
    #     jac = torch.stack(jac)
    #     kernel = jac @ jac.T
    #     print('kernel({}) = {:.2e} +- {:.2e}'.format(module, kernel.mean().item(), kernel.std().item()), flush=True)
    #     lr = lr / (kernel.mean() + kernel.std()).item()
    #     param_groups.append({
    #         'params': list(module.parameters()),
    #         'lr': lr,
    #     })

    # lrs = torch.tensor([x['lr'] for x in param_groups])
    # lrs = config['lr'] * lrs / lrs.max().item()

    # for group, lr in zip(param_groups, lrs):
    #     group['lr'] = lr.item()

    optim = torch.optim.Adam(model.parameters(), lr=config['lr'])
    # print(optim, flush=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=25, factor=0.5, verbose=True)

    dynamics = []
    wall = time.perf_counter()
    wall_print = time.perf_counter()

    for epoch in itertools.count():

        errs = []
        loader = DataLoader(train_dataset, batch_size=config['bs'], shuffle=True)
        for step, data in enumerate(loader):
            data = data.to(device)

            pred = model(data.z, data.pos, data.batch)
            optim.zero_grad()
            (pred.view(-1) - data.y[:, config['target']]).pow(2).mean().backward()
            optim.step()

            err = pred.view(-1) - data.y[:, config['target']]
            errs += [err.cpu().detach()]

            if time.perf_counter() - wall_print > 15:
                wall_print = time.perf_counter()
                w = time.perf_counter() - wall
                e = epoch + (step + 1) / len(loader)
                print((
                    f'[{e:.1f}] ['
                    f'wall={w / 3600:.2f}h '
                    f'wall/epoch={w / e:.0f}s '
                    f'wall/step={1e3 * w / e / len(loader):.0f}ms '
                    f'step={step}/{len(loader)} '
                    f'mae={units * torch.cat(errs)[-200:].abs().mean():.5f} '
                    f'lr={min(x["lr"] for x in optim.param_groups):.1e}-{max(x["lr"] for x in optim.param_groups):.1e}]'
                ), flush=True)

        if epoch == 0:
            with torch.autograd.profiler.profile(use_cuda=torch.cuda.is_available(), record_shapes=True) as prof:
                for step, data in enumerate(loader):
                    data = data.to(device)
                    pred = model(data.z, data.pos, data.batch)
                    mse = (pred.view(-1) - data.y[:, config['target']]).pow(2)
                    mse.mean().backward()
                    if step == 1:
                        break
            prof.export_chrome_trace(f"{datetime.datetime.now()}.json")
            break


        train_err = torch.cat(errs)

        errs = []
        loader = DataLoader(val_dataset, batch_size=256)
        for data in loader:
            data = data.to(device)
            with torch.no_grad():
                pred = model(data.z, data.pos, data.batch)

            err = pred.view(-1) - data.y[:, config['target']]
            errs += [err.cpu().detach()]
        val_err = torch.cat(errs)

        lrs = [
            x['lr']
            for x in optim.param_groups
        ]
        dynamics += [{
            'epoch': epoch,
            'wall': time.perf_counter() - wall,
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
        }]
        dynamics[-1]['_runtime'] = dynamics[-1]['wall']
        wandb.log(dynamics[-1])

        print(f'[{epoch}] Target: {config["target"]:02d}, MAE TRAIN: {units * train_err.abs().mean():.5f} ± {units * train_err.abs().std():.5f}, MAE VAL: {units * val_err.abs().mean():.5f} ± {units * val_err.abs().std():.5f}', flush=True)

        scheduler.step(val_err.pow(2).mean())

        yield {
            'args': config,
            'dynamics': dynamics,
            'state': {k: v.cpu() for k, v in model.state_dict().items()},
        }

        if dynamics[-1]['wall'] > config['wall']:
            break


def main():
    git = {
        'log': subprocess.getoutput('git log --format="%H" -n 1 -z'),
        'status': subprocess.getoutput('git status -z'),
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str)
    parser.add_argument("--mul0", type=int, default=256)
    parser.add_argument("--mul1", type=int, default=16)
    parser.add_argument("--mul2", type=int, default=0)
    parser.add_argument("--lmax", type=int, default=1)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--rad_gaussians", type=int, default=50)
    parser.add_argument("--rad_h", type=int, default=128)
    parser.add_argument("--rad_layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--bs", type=int, default=50)
    parser.add_argument("--target", type=int, default=7)
    parser.add_argument("--mean", type=float, default=0)
    parser.add_argument("--std", type=float, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wall", type=int, default=(3 * 24 - 1) * 3600)

    args = parser.parse_args()

    wandb.login()
    wandb.init(project=f"qm9" + (f" {args.target}" if args.target != 7 else ""), config=args.__dict__)
    config = dict(wandb.config)
    print(config)

    if config['output']:
        with open(config['output'], 'wb') as handle:
            pickle.dump(config, handle)

    for data in execute(config):
        if config['output']:
            data['git'] = git
            with open(config['output'], 'wb') as handle:
                pickle.dump(config, handle)
                pickle.dump(data, handle)


if __name__ == "__main__":
    main()
