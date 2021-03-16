import torch
try:
    from radam import RAdam
except ImportError:
    pass

from .flexible_primitives import FlexiblePrimitivesBuilder, \
    train_on_batch as train_on_batch_with_flexible_primitives, \
    validate_on_batch as validate_on_batch_with_flexible_primitives


class OptimizerWrapper(object):
    def __init__(self, optimizer, aggregate=1):
        self.optimizer = optimizer
        self.aggregate = aggregate
        self._calls = 0

    def zero_grad(self):
        if self._calls == 0:
            self.optimizer.zero_grad()

    def step(self):
        self._calls += 1
        if self._calls == self.aggregate:
            self._calls = 0
            self.optimizer.step()


def optimizer_factory(config, parameters):
    """Based on the provided config create the suitable optimizer."""
    optimizer = config.get("optimizer", "Adam")
    lr = config.get("lr", 1e-3)
    momentum = config.get("momentum", 0.9)
    weight_decay = config.get("weight_decay", 0.0)

    if optimizer == "SGD":
        return OptimizerWrapper(
            torch.optim.SGD(parameters, lr=lr, momentum=momentum,
                            weight_decay=weight_decay),
            config.get("aggregate", 1)
        )
    elif optimizer == "Adam":
        return OptimizerWrapper(
            torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay),
            config.get("aggregate", 1)
        )
    elif optimizer == "RAdam":
        return OptimizerWrapper(
            RAdam(parameters, lr=lr, weight_decay=weight_decay),
            config.get("aggregate", 1)
        )
    else:
        raise NotImplementedError()


def build_network(config, weight_file=None, device="cpu"):
    network, train_on_batch, validate_on_batch = get_network_with_type(config)
    network.to(device)
    # Check whether there is a weight file provided to continue training from
    if weight_file is not None:
        network.load_state_dict(
            torch.load(weight_file, map_location=device)
        )

    return network, train_on_batch, validate_on_batch


def get_network_with_type(config):
    network_type = config["network"]["type"]
    if network_type == "flexible_primitives":
        network = FlexiblePrimitivesBuilder(config).network
        train_on_batch = train_on_batch_with_flexible_primitives
        validate_on_batch = validate_on_batch_with_flexible_primitives
    else:
        raise NotImplementedError()
    return network, train_on_batch, validate_on_batch
