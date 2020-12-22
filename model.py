# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring, bare-except, abstract-method, arguments-differ
from math import pi

import torch
from torch_geometric.nn import radius_graph
from torch_scatter import scatter

from e3nn_core import o3
from e3nn_core.math import swish
from e3nn_core.nn import (FC, WeightedTensorProduct, Gate,
                            GaussianBasis, Linear)

qm9_target_dict = {
    0: 'dipole_moment',
    1: 'isotropic_polarizability',
    2: 'homo',
    3: 'lumo',
    4: 'gap',
    5: 'electronic_spatial_extent',
    6: 'zpve',
    7: 'energy_U0',
    8: 'energy_U',
    9: 'enthalpy_H',
    10: 'free_energy',
    11: 'heat_capacity',
}


def print_std(name, x):
    pass
    # print(f"{name}{list(x.shape)}: {x.mean(0).abs().mean():.2f} +- {x.var(0).mean().sqrt():.1f}")


class Network(torch.nn.Module):
    def __init__(
            self,
            muls=(256, 16, 0),
            lmax=1,
            num_layers=3,
            cutoff=10.0,
            rad_gaussians=50,
            rad_hs=(128, 128),
            num_filters=8,
            num_neighbors=20,
            num_atoms=20,
            mean=None,
            std=None,
            scale=None,
            atomref=None
        ):
        super().__init__()

        self.cutoff = cutoff
        self.mean = mean
        self.std = std
        self.scale = scale
        self.num_neighbors = num_neighbors
        self.num_atoms = num_atoms

        self.z_emb = torch.nn.Parameter(torch.randn(25, rad_hs[-1]))
        self.radial = torch.nn.Sequential(
            GaussianBasis(rad_gaussians, cutoff),
            FC((rad_gaussians, ) + rad_hs, swish, variance_in=1 / rad_gaussians, out_act=True)
        )
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax)  # spherical harmonics representation

        irreps = self.irreps_sh
        modules = []
        for _ in range(num_layers):
            act = make_gated_block(irreps, muls, self.irreps_sh)
            conv = Conv(irreps, act.irreps_in, self.irreps_sh, 2 * rad_hs[-1], num_filters)
            irreps = act.irreps_out.simplify()

            modules += [torch.nn.ModuleList([conv, act])]

        self.layers = torch.nn.ModuleList(modules)

        self.irreps_out = o3.Irreps("0e + 0o")
        self.layers.append(Conv(irreps, self.irreps_out, self.irreps_sh, 2 * rad_hs[-1], num_filters))

        self.register_buffer('atomref', atomref)

    def forward(self, z, pos, batch=None):
        assert z.dim() == 1 and z.dtype == torch.long
        assert pos.dim() == 2 and pos.shape[1] == 3
        batch = torch.zeros_like(z) if batch is None else batch

        edge_src, edge_dst = radius_graph(pos, r=self.cutoff, batch=batch, max_num_neighbors=1000)
        edge_vec = pos[edge_src] - pos[edge_dst]
        edge_sh = o3.spherical_harmonics(self.irreps_sh, edge_vec, 'component')
        edge_len = edge_vec.norm(dim=1)
        edge_len_emb = self.radial(edge_len)
        edge_c = (pi * edge_len / self.cutoff).cos().add(1).div(2)
        edge_sh = edge_c[:, None] * edge_sh / self.num_neighbors**0.5

        # z : [1, 6, 7, 8, 9] -> [0, 1, 2, 3, 4]
        index_z = z.new_tensor([-1, 0, -1, -1, -1, -1, 1, 2, 3, 4])[z]
        edge_z = 5 * index_z[edge_src] + index_z[edge_dst]
        edge_z_emb = self.z_emb[edge_z]

        edge_query = torch.cat([edge_z_emb, edge_len_emb], dim=1)

        h = scatter(edge_sh, edge_src, dim=0, dim_size=len(pos))
        h[:, 0] = 1

        print_std('h', h)
        print_std('edge_query', edge_query)

        for conv, act in self.layers[:-1]:
            with torch.autograd.profiler.record_function("Layer"):
                h = conv(h, edge_src, edge_dst, edge_query, edge_sh)  # convolution
                print_std('post conv', h)
                h = act(h)  # gate non linearity
                print_std('post gate', h)

        with torch.autograd.profiler.record_function("Layer"):
            h = self.layers[-1](h, edge_src, edge_dst, edge_query, edge_sh)

        print_std('h out', h)

        s = 0
        for i, (mul, (l, p)) in enumerate(self.irreps_out):
            assert mul == 1 and l == 0
            if p == 1:
                s += h[:, i]
            if p == -1:
                s += h[:, i].pow(2).mul(0.5)  # odd^2 = even
        h = s.view(-1, 1)

        print_std('h out+', h)

        # for the scatter we normalize
        h = h / self.num_atoms**0.5

        if self.mean is not None and self.std is not None:
            h = h * self.std + self.mean

        if self.atomref is not None:
            h = h + self.atomref[z]

        out = scatter(h, batch, dim=0)

        if self.scale is not None:
            out = self.scale * out

        return out


def make_gated_block(irreps_in, muls, irreps_sh):
    """
    Make a Gate assuming many things
    """
    irreps_available = [
        (l, p_in * p_sh)
        for _, (l_in, p_in) in irreps_in.simplify()
        for _, (l_sh, p_sh) in irreps_sh
        for l in range(abs(l_in - l_sh), l_in + l_sh + 1)
    ]

    scalars = o3.Irreps([(muls[0], 0, p) for p in (1, -1) if (0, p) in irreps_available])
    act_scalars = [(mul, swish if p == 1 else torch.tanh) for mul, (_, p) in scalars]

    nonscalars = o3.Irreps([(muls[l], l, p*(-1)**l) for l in range(1, len(muls)) for p in (1, -1) if (l, p*(-1)**l) in irreps_available])
    if (0, +1) in irreps_available:
        gates = o3.Irreps([(nonscalars.num_irreps, 0, +1)])
        act_gates = [(-1, torch.sigmoid)]
    else:
        gates = o3.Irreps([(nonscalars.num_irreps, 0, -1)])
        act_gates = [(-1, torch.tanh)]

    return Gate(scalars, act_scalars, gates, act_gates, nonscalars)


class Conv(torch.nn.Module):
    def __init__(self, irreps_in, irreps_out, irreps_sh, dim_key, num_filters):
        super().__init__()
        self.irreps_in = irreps_in.simplify()
        self.irreps_out = irreps_out.simplify()
        self.irreps_sh = irreps_sh.simplify()

        self.si = Linear(self.irreps_in, self.irreps_out, internal_weights=True, shared_weights=True)
        # self.si_weight = torch.nn.Parameter(torch.randn(5, self.si.tp.weight_numel))

        self.lin1 = Linear(self.irreps_in, self.irreps_in, internal_weights=True, shared_weights=True)
        # self.lin1_weight = torch.nn.Parameter(torch.randn(5, self.lin1.tp.weight_numel))

        instr = []
        irreps = []
        for i_1, (mul_1, (l_1, p_1)) in enumerate(self.irreps_in):
            for i_2, (_, (l_2, p_2)) in enumerate(self.irreps_sh):
                for l_out in range(abs(l_1 - l_2), l_1 + l_2 + 1):
                    p_out = p_1 * p_2
                    if (l_out, p_out) in [(l, p) for _, (l, p) in self.irreps_out]:
                        r = (mul_1, l_out, p_out)
                        if r in irreps:
                            i_out = irreps.index(r)
                        else:
                            i_out = len(irreps)
                            irreps.append(r)
                        instr += [(i_1, i_2, i_out, 'uvu', True, 1.0)]
        irreps = o3.Irreps(irreps)
        in1 = [(mul, ir, 1.0) for mul, ir in self.irreps_in]
        in2 = [(mul, ir, 1.0) for mul, ir in self.irreps_sh]
        out = [(mul, ir, 1.0) for mul, ir in irreps]
        self.tp = WeightedTensorProduct(in1, in2, out, instr, internal_weights=False, shared_weights=False)

        self.tp_keys = torch.nn.Parameter(torch.randn(num_filters, dim_key))
        self.tp_weight = torch.nn.Parameter(torch.randn(num_filters, self.tp.weight_numel))

        self.lin2 = Linear(irreps, self.irreps_out, internal_weights=True, shared_weights=True)
        # self.lin2_weight = torch.nn.Parameter(torch.randn(5, self.lin2.tp.weight_numel))

    def forward(self, x, edge_src, edge_dst, edge_query, edge_sh):
        with torch.autograd.profiler.record_function("Conv"):
            # x = [num_atoms, dim(irreps_in)]
            s = self.si(x)

            x = self.lin1(x)

            weight = torch.softmax(edge_query @ self.tp_keys.T / self.tp_keys.shape[1]**0.5, dim=1) @ self.tp_weight

            # edge_sh are divided by sqrt(num_neighbors)
            edge_x = self.tp(x[edge_src], edge_sh, weight)
            x = scatter(edge_x, edge_dst, dim=0, dim_size=len(x))

            x = self.lin2(x)

            print_std('self', s)
            print_std('+x', x)
            return s + x / 10
