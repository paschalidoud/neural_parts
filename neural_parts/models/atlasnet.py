from collections import namedtuple
from itertools import product

import torch
from torch import nn

from ..external.libmesh import check_mesh_contains
from ..utils import sphere_mesh

_Mesh = namedtuple("Mesh", ["vertices", "faces"])


class AtlasNet(nn.Module):
    def __init__(self, feature_dims, radius=0.25):
        super().__init__()
        self.radius = radius
        self.map_to_feature_dims = nn.Conv1d(3, feature_dims, 1)

        dims = [feature_dims, 1024, 512, 256, 128, 3]
        self.linear_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for a, b in zip(dims, dims[1:]):
            self.batch_norms.append(nn.BatchNorm1d(a))
            self.linear_layers.append(nn.Conv1d(a, b, 1))

    def _check_shapes(self, F, x):
        B1, M1, _ = F.shape
        B2, _, M2, D = x.shape
        assert B1 == B2 and M1 == M2 and D == 3

    def _merge_batch_dims(self, F, x):
        B, M, D = F.shape
        _, N, _, _ = x.shape
        F = F.reshape(-1, D)
        x = x.permute(0, 2, 3, 1).reshape(-1, 3, N)

        return F, x, B, M

    def _split_batch_dims(self, x, B, M):
        x = x.view(B, M, 3, -1)
        return x.permute(0, 3, 1, 2).contiguous()

    def forward(self, F, x):
        self._check_shapes(F, x)
        F, x, B, M = self._merge_batch_dims(F, x)

        # Add the image features to the point features
        x = self.map_to_feature_dims(x) + F[:, :, None]

        # Transform the points
        for b, l in zip(self.batch_norms, self.linear_layers):
            x = l(b(x).relu())

        return torch.tanh(self._split_batch_dims(x, B, M))

    @torch.no_grad()
    def inverse(self, F, y):
        B, N, M, _ = y.shape

        # Compute the meshes
        v, faces = sphere_mesh(200, self.radius)
        v = v[None, :, None].expand(B, -1, M, -1)
        v_prime = self(F, v)

        # Cast everything to numpy instead of pytorch
        v_prime = v_prime.numpy()
        faces = faces.numpy()
        y = y.numpy()

        # Compute whether the points are internal to the meshes
        output = torch.zeros(B, N, M, 3)
        for b, m in product(range(B), range(M)):
            output[b, :, m, 0] = torch.from_numpy((~check_mesh_contains(
                _Mesh(v_prime[b, :, m], faces),
                y[b, :, m]
            )).astype("float32") * 2 * self.radius)

        return output, None
