from networks.actor_critic_network import ContinuousActorCritic, layer_init
from networks.discrete_actor_critic import DiscreteActorCritic
from networks.normalization import ObsNormalizer, RunningMeanStd
from networks.q_network import QNetwork

__all__ = ["ContinuousActorCritic", "DiscreteActorCritic", "layer_init", "ObsNormalizer",
           "RunningMeanStd", "QNetwork", "get_network"]


def get_network(name: str) -> type:
    """The network class a config's ``network:`` key names.

    A plain name is a class exported above (``"QNetwork"``); a dotted path
    imports it from its own module (``"networks.my_net.MyNet"``) — handy for
    files you keep out of this list.  Every algorithm builds its network as
    ``get_network(args.network)(envs, **args.network_kwargs)``.
    """
    if "." in name:
        import importlib

        module_path, _, cls_name = name.rpartition(".")
        return getattr(importlib.import_module(module_path), cls_name)
    try:
        return globals()[name]
    except KeyError:
        raise ValueError(f"unknown network {name!r}: export it from networks/__init__.py "
                         f"or give its dotted import path (available: {', '.join(__all__)})") from None
