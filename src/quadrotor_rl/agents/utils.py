import torch
import torch.nn as nn


def hard_update(
    target_network: nn.Module,
    source_network: nn.Module,
) -> None:
    """
    Copy all parameters from source_network to target_network.

    Used when target networks are initialized.
    """

    target_network.load_state_dict(
        source_network.state_dict()
    )


@torch.no_grad()
def polyak_update(
    target_network: nn.Module,
    source_network: nn.Module,
    tau: float,
) -> None:
    """
    Slowly move target-network parameters toward source-network parameters.

    target <- (1 - tau) * target + tau * source
    """

    if not 0.0 <= tau <= 1.0:
        raise ValueError(
            f"tau must be in [0, 1], got {tau}."
        )

    for target_param, source_param in zip(
        target_network.parameters(),
        source_network.parameters(),
    ):
        target_param.data.mul_(
            1.0 - tau
        )

        target_param.data.add_(
            tau * source_param.data
        )