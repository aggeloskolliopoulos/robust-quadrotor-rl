import torch
import torch.nn as nn
from torch.distributions import Normal


class Critic(nn.Module):
    """Q-function approximator for continuous state-action spaces."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()

        # -------------------------------------------------------------
        # The critic receives BOTH the state and the action.
        #
        # For our quadrotor:
        #
        #     observation_dim = 16
        #     action_dim      = 4
        #
        # so the critic input dimension is 20.
        # -------------------------------------------------------------
        input_dim = observation_dim + action_dim

        # -------------------------------------------------------------
        # Simple multi-layer perceptron:
        #
        #     (state, action)
        #          ↓
        #        256
        #          ↓
        #        256
        #          ↓
        #       Q(s, a)
        #
        # The final layer has ONE output because Q(s, a) is a scalar.
        # -------------------------------------------------------------
        self.network = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim,
            ),
            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                1,
            ),
        )

    def forward(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate Q(s, a) for a batch of state-action pairs."""

        # -------------------------------------------------------------
        # Concatenate state and action along the feature dimension.
        #
        # Example batch:
        #
        #     observation.shape = [256, 16]
        #     action.shape      = [256, 4]
        #
        # After concatenation:
        #
        #     x.shape           = [256, 20]
        # -------------------------------------------------------------
        x = torch.cat(
            [observation, action],
            dim=-1,
        )

        q_value = self.network(x)

        return q_value

class GaussianActor(nn.Module):
    """Gaussian stochastic policy for Soft Actor-Critic."""

    LOG_STD_MIN = -20.0
    LOG_STD_MAX = 2.0

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()

        # -------------------------------------------------------------
        # Shared neural-network backbone.
        #
        # observation
        #     ↓
        # hidden representation
        # -------------------------------------------------------------
        self.backbone = nn.Sequential(
            nn.Linear(
                observation_dim,
                hidden_dim,
            ),
            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.ReLU(),
        )

        # -------------------------------------------------------------
        # Gaussian distribution parameters.
        #
        # For our quadrotor:
        #
        #     mean    ∈ R^4
        #     log_std ∈ R^4
        # -------------------------------------------------------------
        self.mean_layer = nn.Linear(
            hidden_dim,
            action_dim,
        )

        self.log_std_layer = nn.Linear(
            hidden_dim,
            action_dim,
        )

    def forward(
        self,
        observation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the Gaussian mean and log standard deviation."""

        features = self.backbone(
            observation
        )

        mean = self.mean_layer(
            features
        )

        log_std = self.log_std_layer(
            features
        )

        # Keep the Gaussian variance numerically well behaved.
        log_std = torch.clamp(
            log_std,
            self.LOG_STD_MIN,
            self.LOG_STD_MAX,
        )

        return mean, log_std

    def sample(
        self,
        observation: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Sample a bounded SAC action using the reparameterization trick.

        Returns:
            action:
                Stochastic action in (-1, 1).

            log_prob:
                Log probability of the squashed action.

            deterministic_action:
                tanh(mean), used for deterministic evaluation.
        """

        # -------------------------------------------------------------
        # 1. Get Gaussian parameters from the neural network.
        # -------------------------------------------------------------
        mean, log_std = self.forward(
            observation
        )

        std = torch.exp(
            log_std
        )

        # -------------------------------------------------------------
        # 2. Define the Gaussian policy distribution.
        #
        #     u ~ N(mean, std^2)
        # -------------------------------------------------------------
        distribution = Normal(
            mean,
            std,
        )

        # -------------------------------------------------------------
        # 3. Reparameterized sampling.
        #
        # Conceptually:
        #
        #     epsilon ~ N(0, I)
        #
        #     u = mean + std * epsilon
        #
        # rsample(), unlike sample(), preserves the gradient path
        # through mean and std.
        # -------------------------------------------------------------
        pre_tanh_action = distribution.rsample()

        # -------------------------------------------------------------
        # 4. Squash the unbounded Gaussian sample into (-1, 1).
        #
        # This matches our environment's normalized action space.
        # -------------------------------------------------------------
        action = torch.tanh(
            pre_tanh_action
        )

        # -------------------------------------------------------------
        # 5. Gaussian log probability before tanh.
        #
        # Shape:
        #
        #     [batch_size, action_dim]
        # -------------------------------------------------------------
        log_prob = distribution.log_prob(
            pre_tanh_action
        )

        # -------------------------------------------------------------
        # 6. Correct the log probability for the tanh transformation.
        #
        #     da/du = 1 - tanh(u)^2
        #
        # therefore the probability density must include the
        # corresponding Jacobian correction.
        # -------------------------------------------------------------
        log_prob -= torch.log(
            1.0 - action.pow(2) + 1e-6
        )

        # -------------------------------------------------------------
        # 7. Sum over the action dimensions.
        #
        # Our 4 motor actions form one joint action, so SAC needs one
        # scalar log probability per state.
        #
        #     [batch_size, 4]
        #              ↓
        #     [batch_size, 1]
        # -------------------------------------------------------------
        log_prob = log_prob.sum(
            dim=-1,
            keepdim=True,
        )

        # -------------------------------------------------------------
        # 8. Deterministic policy action.
        #
        # During evaluation we use the center of the Gaussian without
        # sampling exploration noise.
        # -------------------------------------------------------------
        deterministic_action = torch.tanh(
            mean
        )

        return (
            action,
            log_prob,
            deterministic_action,
        )