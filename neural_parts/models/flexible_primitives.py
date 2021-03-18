from functools import partial

import torch
from torch import nn

from .atlasnet import AtlasNet
from .feature_extractors import get_feature_extractor
from .projection_layer import get_projection_layer
from .simple_nvp import SimpleNVP
from ..utils import sphere_mesh, sphere_edges


class FlexiblePrimitives(nn.Module):
    def __init__(self, radius, n_primitives, input_dims, feature_extractor,
                 invertible_network, with_dropout=False):
        super().__init__()
        self.radius = radius
        self.n_primitives = n_primitives
        self.input_dims = input_dims

        self.feature_extractor = feature_extractor
        self.invertible_network = invertible_network
        feature_dims = feature_extractor.feature_size

        self.with_dropout = with_dropout
        if self.with_dropout:
            self.dropout = nn.Dropout(p=0.5)

        self.register_parameter(
            "primitive_embedding",
            torch.nn.Parameter(
                torch.randn(n_primitives, feature_extractor.feature_size)
            )
        )

    def compute_features(self, X):
        """Compute the per primitive features for the input image X."""
        F = self.feature_extractor(X)
        if self.with_dropout:
            F = self.dropout(F)
        F = F[:, None].expand(-1, self.n_primitives, -1)
        F = torch.cat([
            F,
            self.primitive_embedding[None].expand_as(F)
        ], dim=-1)

        B = F.shape[0]
        M = self.n_primitives
        D = 2*self.feature_extractor.feature_size

        assert F.shape == (B, M, D)
        return F

    def implicit_surface(self, F, y):
        """Compute the implicit surface for the points y given per primitive
        features F."""
        y = y[:, :, None].expand(-1, -1, self.n_primitives, -1)
        y_latent, ldj = self.invertible_network.inverse(F, y)
        norm = torch.sqrt((y_latent**2).sum(-1))

        # <0 denotes internal points
        # >0 denotes external points
        #  0 is the boundary hence our primitive
        return norm - self.radius, ldj

    def euclidean_implicit_surface(self, F, y, S=200):
        phi, ldj = self.implicit_surface(F, y)
        y_prim = self.points_on_primitives(F, S=S, random=True, union_surface=False)

        diff = y[:, None, :, None] - y_prim[:, :, None, :]
        dists, _ = torch.sqrt((diff**2).sum(-1)).min(1)

        return dists * torch.sign(phi), ldj

    def points_on_primitives(self, F, S=200, random=True, union_surface=True,
                             mesh=False, edges=False):
        """Compute points on the primitives in world coordinates given per
        primitive features.

        Arguments
        ---------
            F: Tensor of shape (B, M, D) that contains a feature per primitive
            S: int, how many points to sample per primitive
            random: bool, if set to True then sample uniformly in the latent
                    space of each primitive (aka on the sphere)
            union_surface: bool, if True then compute a boolean mask that
                           indicates which of the returned points is actually
                           on the surface of the union of the primitives,
                           namely discarding points that are on the surface of
                           one primitive and inside another.
            mesh: bool, if True use pre-triangulated points and return the
                  faces to create a mesh for each primitive
            edges: bool, if True use pre-triangulated points that are
                   approximately equidistant and return the edges to be able to
                   compute edge lengths
        """
        # Extract some shapes
        B = F.shape[0]
        M = self.n_primitives

        # Sample the points on the primitives (either randomly or triangulated)
        if random:
            y = torch.randn(B, S, M, self.input_dims, device=F.device)
            y /= torch.sqrt((y**2).sum(-1, keepdim=True)) / self.radius
        elif mesh:
            assert self.input_dims == 3
            y, faces = sphere_mesh(S, self.radius, F.device)
            y = y[None, :, None].expand(B, -1, M, -1)
        elif edges:
            y, _edges = sphere_edges(divisions=2, radius=self.radius,
                                     device=F.device)
            y = y[None, :, None].expand(B, -1, M, -1)

        # Compute the points on each primitive
        y_prim = self.invertible_network(F, y)

        # If requested discard points that are internal to another primitive
        if union_surface:
            assert not mesh  # it could be made to work but it might become
                             # meshy :-P
            phi_y_prim, _ = self.implicit_surface(F, y_prim.view(B, S*M, -1))
            phi_y_prim = phi_y_prim.view(B, S, M, M)
            on_surface = (phi_y_prim >= 0).all(dim=-1)

        # Return what was requested
        if mesh:
            return y_prim, faces
        if edges:
            return y_prim, _edges
        if union_surface:
            return y_prim, on_surface
        return y_prim


class FlexiblePrimitivesBuilder(object):
    def __init__(self, config):
        self.config = config
        self._feature_extractor = None
        self._invertible_network = None
        self._network = None

    @property
    def feature_extractor(self):
        if self._feature_extractor is None:
            self._feature_extractor = get_feature_extractor(
                self.config["feature_extractor"]["type"],
                self.config["feature_extractor"].get("freeze_bn", False),
                self.config["feature_extractor"].get("feature_size", 128)
            )
        return self._feature_extractor

    @property
    def invertible_network(self):
        if self._invertible_network is None:
            networks = {
                "simple_nvp": partial(
                    SimpleNVP,
                    self.config["invertible_network"]["n_blocks"],
                    2*self.feature_extractor.feature_size,
                    self.config["invertible_network"]["hidden_size"],
                    get_projection_layer(**self.config["projection_layer"]),
                    self.config["invertible_network"]["checkpoint"],
                    self.config["invertible_network"]["normalize"],
                    self.config["invertible_network"]["explicit_affine"],
                ),
                "atlasnet": partial(
                    AtlasNet,
                    2*self.feature_extractor.feature_size
                )
            }
            type = self.config["invertible_network"]["type"]
            self._invertible_network = networks[type]()
        return self._invertible_network

    @property
    def network(self):
        if self._network is None:
            self._network = FlexiblePrimitives(
                self.config["network"]["sphere_radius"],
                self.config["network"]["n_primitives"],
                self.config["data"]["input_dims"],
                self.feature_extractor,
                self.invertible_network,
                self.config["network"]["with_dropout"],
            )
        return self._network


def compute_predictions_from_features(F, network, targets, config):
    # Dictionary to keep the predictions
    predictions = {}

    # Compute the implicit surface value per primitive per point in space if
    # needed
    if config["network"].get("phi_volume", False):
        predictions["phi_volume"], predictions["ldj_volume"] = \
            network.implicit_surface(F, targets[0])

    # Compute the euclidean implicit surface value per primitive per point in
    # space if needed
    # NOTE: It overrides the plain 'phi_volume' for backwards compatibility
    if config["network"].get("euclidean_phi_volume", False):
        predictions["phi_volume"], predictions["ldj_volume"] = \
            network.euclidean_implicit_surface(
                F,
                targets[0],
                S=config["network"].get("n_points_on_sphere", 200)
            )

    # Compute the implicit surface value per primitive per point on the target
    # surface if needed
    if config["network"].get("phi_surface", False):
        predictions["phi_surface"], predictions["ldj_surface"] = \
            network.implicit_surface(F, targets[-1][:, :, :3])

    # Compute points on the predicted surface if needed
    if config["network"].get("y_prim", False):
        if config["network"].get("union_surface", True):
            y_prim, on_surface = network.points_on_primitives(
                F,
                config["network"].get("n_points_on_sphere", 200),
                union_surface=True
            )
            predictions["y_prim"] = y_prim
            predictions["on_surface"] = on_surface
        else:
            predictions["y_prim"] = network.points_on_primitives(
                F,
                config["network"].get("n_points_on_sphere", 200),
                union_surface=False
            )

    # Compute points on the predicted surface together with edge lengths
    if config["network"].get("edge_lengths", False):
        y_edge, edges = network.points_on_primitives(
            F,
            0,  # S is ignored so let it be 0
            random=False,
            union_surface=False,
            edges=True
        )
        predictions["y_edge"] = y_edge
        predictions["edges"] = edges

    return predictions


def train_on_batch(network, optimizer, loss_fn, metrics_fn, X, targets,
                   config):
    """Perform a forward and backward pass on a batch of samples and compute
    the loss.
    """
    optimizer.zero_grad()

    # Extract the per primitive features
    F = network.compute_features(X)
    predictions = compute_predictions_from_features(
        F, network, targets, config
    )

    # Do the forward pass to predict the primitive_parameters
    batch_loss = loss_fn(predictions, targets, config["loss"])
    metrics_fn(predictions, targets)
    # Do the backpropagation
    batch_loss.backward()
    nn.utils.clip_grad_norm_(network.parameters(), 1)
    # Do the update
    optimizer.step()

    return batch_loss.item()


def validate_on_batch(network, loss_fn, metrics_fn, X, targets, config):
    # Extract the per primitive features
    F = network.compute_features(X)
    predictions = compute_predictions_from_features(
        F, network, targets, config
    )
    # Do the forward pass to predict the primitive_parameters
    batch_loss = loss_fn(predictions, targets, config["loss"])
    metrics_fn(predictions, targets)

    return batch_loss.item()
