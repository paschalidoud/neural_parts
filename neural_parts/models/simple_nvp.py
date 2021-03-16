import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


from ..utils import quaternions_to_rotation_matrices


class CouplingLayer(nn.Module):
    def __init__(self, map_s, map_t, projection, mask):
        super().__init__()
        self.map_s = map_s
        self.map_t = map_t
        self.projection = projection
        self.register_buffer("mask", mask)

    def forward(self, F, y):
        y1 = y * self.mask

        F_y1 = torch.cat([F, self.projection(y1)], dim=-1)
        s = self.map_s(F_y1)
        t = self.map_t(F_y1)

        x = y1 + (1-self.mask) * ((y - t) * torch.exp(-s))
        ldj = (-s).sum(-1)

        return x, ldj

    def inverse(self, F, x):
        x1 = x * self.mask

        F_x1 = torch.cat([F, self.projection(x1)], dim=-1)
        s = self.map_s(F_x1)
        t = self.map_t(F_x1)

        y = x1 + (1-self.mask) * (x * torch.exp(s) + t)
        ldj = s.sum(-1)

        return y, ldj


class SimpleNVP(nn.Module):
    def __init__(
        self, n_layers, feature_dims, hidden_size, projection,
        checkpoint=True, normalize=True, explicit_affine=True
    ):
        super().__init__()
        self._checkpoint = checkpoint
        self._normalize = normalize
        self._explicit_affine = explicit_affine
        self._projection = projection
        self._create_layers(n_layers, feature_dims, hidden_size)

    def _create_layers(self, n_layers, feature_dims, hidden_size):
        input_dims = 3
        proj_dims = self._projection.proj_dims

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            mask = torch.zeros(input_dims)
            mask[torch.randperm(input_dims)[:2]] = 1

            map_s = nn.Sequential(
                nn.Linear(proj_dims+feature_dims, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, input_dims),
                nn.Hardtanh(min_val=-10, max_val=10)
            )
            map_t = nn.Sequential(
                nn.Linear(proj_dims+feature_dims, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, input_dims)
            )
            self.layers.append(CouplingLayer(
                map_s,
                map_t,
                self._projection,
                mask[None, None, None]
            ))

        if self._normalize:
            self.scales = nn.Sequential(
                nn.Linear(feature_dims, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 3)
            )

        if self._explicit_affine:
            self.rotations = nn.Sequential(
                nn.Linear(feature_dims, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 4)
            )
            self.translations = nn.Sequential(
                nn.Linear(feature_dims, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 3)
            )

    def _check_shapes(self, F, x):
        B1, M1, _ = F.shape
        B2, _, M2, D = x.shape
        assert B1 == B2 and M1 == M2 and D == 3

    def _expand_features(self, F, x):
        _, N, _, _ = x.shape
        return F[:, None].expand(-1, N, -1, -1)

    def _call(self, func, *args, **kwargs):
        if self._checkpoint:
            return checkpoint(func, *args, **kwargs)
        else:
            return func(*args, **kwargs)

    def _normalize_input(self, F, y):
        if not self._normalize:
            return 0, 1

        sigma = torch.nn.functional.elu(self.scales(F)) + 1
        sigma = sigma[:, None]

        return 0, sigma

    def _affine_input(self, F, y):
        if not self._explicit_affine:
            return torch.eye(3)[None, None, None], 0

        q = self.rotations(F)
        q = q / torch.sqrt(torch.square(q).sum(-1, keepdim=True))
        R = quaternions_to_rotation_matrices(q[:, None])
        t = self.translations(F)[:, None]

        return R, t

    def forward(self, F, x):
        self._check_shapes(F, x)
        mu, sigma = self._normalize_input(F, x)
        R, t = self._affine_input(F, x)
        F = self._expand_features(F, x)

        y = x
        ldj = 0
        for l in self.layers:
            y, ldji = self._call(l, F, y)
            ldj = ldj + ldji
        y = y / sigma + mu
        y = torch.matmul(y.unsqueeze(-2), R).squeeze(-2) + t
        return y

    def inverse(self, F, y):
        self._check_shapes(F, y)
        mu, sigma = self._normalize_input(F, y)
        R, t = self._affine_input(F, y)
        F = self._expand_features(F, y)

        x = y
        x = torch.matmul(
            (x - t).unsqueeze(-2), R.transpose(-2, -1)
        ).squeeze(-2)
        x = (x - mu) * sigma
        ldj = 0
        for l in reversed(self.layers):
            x, ldji = self._call(l.inverse, F, x)
            ldj = ldj + ldji
        return x, ldj
