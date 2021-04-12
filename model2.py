import torch
from e3nn import o3
from e3nn.math import soft_one_hot_linspace
from torch_cluster import radius_graph
from torch_scatter import scatter

from e3nn.nn.models.v2103.gate_points_message_passing import MessagePassing


class Network(torch.nn.Module):
    def __init__(
        self,
        muls=(256, 16, 0),
        lmax=1,
        num_layers=3,
        max_radius=10.0,
        number_of_basis=50,
        fc_neurons=[128, 128],
        num_neighbors=20,
        num_nodes=20,
        mean=None,
        std=None,
        scale=None,
        atomref=None,
    ) -> None:
        super().__init__()

        self.lmax = lmax
        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.num_nodes = num_nodes

        self.mean = mean
        self.std = std
        self.scale = scale

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
            irreps_edge_attr=o3.Irreps.spherical_harmonics(lmax),
            layers=num_layers,
            fc_neurons=[self.number_of_basis] + fc_neurons,
            num_neighbors=num_neighbors,
        )

    def forward(self, node_atom, node_pos, batch) -> torch.Tensor:
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
            l=range(self.lmax + 1),
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
            self.number_of_basis,
            basis='cosine',  # the cosine basis with cutoff = True goes to zero at max_radius
            cutoff=True,  # no need for an additional smooth cutoff
        ).mul(self.number_of_basis**0.5)

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

        if self.mean is not None and self.std is not None:
            node_outputs = node_outputs * self.std + self.mean

        if self.atomref is not None:
            node_outputs = node_outputs + self.atomref[node_atom]
        # for target=7, MAE of 75eV

        outputs = scatter(node_outputs, batch, dim=0)

        if self.scale is not None:
            outputs = self.scale * outputs

        return outputs
