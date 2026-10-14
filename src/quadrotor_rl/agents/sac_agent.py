import math
import numpy as np
import torch
import torch.nn.functional as F

from quadrotor_rl.agents.networks import Critic, GaussianActor
from quadrotor_rl.agents.utils import hard_update, polyak_update


class SACAgent:
    """Soft Actor-Critic agent, built incrementally for learning purposes."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        gamma: float = 0.99,
        tau: float = 0.005,
        initial_alpha: float = 0.2,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        device: str = "cpu",
    ):
        self.device = torch.device(device)

        self.gamma = gamma
        self.tau = tau

        if not 0.0 <= tau <= 1.0:
            raise ValueError(
                f"tau must be in [0, 1], got {tau}."
            )

        # -------------------------------------------------------------
        # Automatic entropy tuning.
        #
        # We optimize log(alpha), not alpha directly, so that:
        #
        #     alpha = exp(log_alpha) > 0
        # -------------------------------------------------------------
        if initial_alpha <= 0.0:
            raise ValueError(
                f"initial_alpha must be positive, got {initial_alpha}."
            )

        self.log_alpha = torch.tensor(
            math.log(initial_alpha),
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )

        # Target entropy heuristic.
        #
        # For our 4-dimensional action space:
        #
        #     target_entropy = -4

        self.target_entropy = -float(
            action_dim
        )

        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha],
            lr=alpha_lr,
        )

        # -------------------------------------------------------------
        # Current stochastic policy.
        # -------------------------------------------------------------
        self.actor = GaussianActor(
            observation_dim=observation_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=actor_lr,
        )

        # -------------------------------------------------------------
        # Two independent online critics.
        #
        # These are the Q-functions that will be trained directly
        # using gradient descent.
        # -------------------------------------------------------------
        self.critic1 = Critic(
            observation_dim=observation_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)

        self.critic2 = Critic(
            observation_dim=observation_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)

        # -------------------------------------------------------------
        # Two target critics.
        #
        # They have the same architecture, but move slowly toward
        # the online critics instead of being optimized directly.
        # -------------------------------------------------------------
        self.target_critic1 = Critic(
            observation_dim=observation_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)

        self.target_critic2 = Critic(
            observation_dim=observation_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)
        
        # Target critics start identical to online critics.
        hard_update(
            target_network=self.target_critic1,
            source_network=self.critic1,
        )

        hard_update(
            target_network=self.target_critic2,
            source_network=self.critic2,
        )

        # -------------------------------------------------------------
        # Target critics are never optimized by gradient descent.
        # -------------------------------------------------------------
        for parameter in self.target_critic1.parameters():
            parameter.requires_grad_(False)

        for parameter in self.target_critic2.parameters():
            parameter.requires_grad_(False)

        # -------------------------------------------------------------
        # Separate optimizers for the two critics.
        # -------------------------------------------------------------
        self.critic1_optimizer = torch.optim.Adam(
            self.critic1.parameters(),
            lr=critic_lr,
        )

        self.critic2_optimizer = torch.optim.Adam(
            self.critic2.parameters(),
            lr=critic_lr,
        )

    @property
    def alpha(self) -> torch.Tensor:
        """Current entropy coefficient."""
        return self.log_alpha.exp()

    def select_action(
        self,
        observation: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
        """
        Select one normalized action for the environment.

        During training:
            deterministic=False
            -> sample from the stochastic SAC policy.

        During evaluation:
            deterministic=True
            -> use tanh(mean).
        """

        observation = np.asarray(
            observation,
            dtype=np.float32,
        )

        # -------------------------------------------------------------
        # Environment gives us one observation:
        #
        #     [observation_dim]
        #
        # Neural networks expect a batch:
        #
        #     [batch_size, observation_dim]
        #
        # so we temporarily add a batch dimension.
        # -------------------------------------------------------------
        observation_tensor = torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        # -------------------------------------------------------------
        # Action selection is not a training operation.
        #
        # We only want a numerical action to send to the environment,
        # so there is no reason to build a gradient graph.
        # -------------------------------------------------------------
        with torch.no_grad():
            (
                stochastic_action,
                _,
                deterministic_action,
            ) = self.actor.sample(
                observation_tensor
            )

        if deterministic:
            action = deterministic_action
        else:
            action = stochastic_action

        # Remove batch dimension:
        #
        #     [1, action_dim]
        #          ↓
        #     [action_dim]
        action = action.squeeze(0)

        # Move back to CPU/NumPy because Gymnasium expects NumPy actions.
        return action.cpu().numpy()
    
    def update_critics(
        self,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, float]:
        """Perform one SAC critic update."""

        observations = batch["observations"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_observations = batch["next_observations"]
        terminated = batch["terminated"]

        # -------------------------------------------------------------
        # Construct the SAC Bellman target.
        #
        # We do NOT want gradients flowing through:
        #
        #     next actor action
        #     target critics
        #     Bellman target
        #
        # The target is treated as a fixed regression target during
        # this critic update.
        # -------------------------------------------------------------
        with torch.no_grad():

            # a' ~ pi(. | s')
            (
                next_actions,
                next_log_prob,
                _,
            ) = self.actor.sample(
                next_observations
            )

            # Target Q estimates.
            target_q1 = self.target_critic1(
                next_observations,
                next_actions,
            )

            target_q2 = self.target_critic2(
                next_observations,
                next_actions,
            )

            # Conservative twin-critic estimate.
            min_target_q = torch.minimum(
                target_q1,
                target_q2,
            )

            # Soft state value:
            #
            # V(s') ≈ min(Q1', Q2') - alpha * log pi(a' | s')
            soft_next_value = (
                min_target_q
                - self.alpha * next_log_prob
            )

            # Bellman target:
            #
            # y = r + gamma * (1-d) * V(s')
            target_q = (
                rewards
                + self.gamma
                * (1.0 - terminated)
                * soft_next_value
            )

        # -------------------------------------------------------------
        # Current critics' predictions for the actions that were
        # actually stored in the replay buffer.
        # -------------------------------------------------------------
        current_q1 = self.critic1(
            observations,
            actions,
        )

        current_q2 = self.critic2(
            observations,
            actions,
        )

        # -------------------------------------------------------------
        # Mean-squared Bellman error.
        # -------------------------------------------------------------
        critic1_loss = F.mse_loss(
            current_q1,
            target_q,
        )

        critic2_loss = F.mse_loss(
            current_q2,
            target_q,
        )

        # -------------------------------------------------------------
        # Update critic 1.
        # -------------------------------------------------------------
        self.critic1_optimizer.zero_grad()

        critic1_loss.backward()

        self.critic1_optimizer.step()

        # -------------------------------------------------------------
        # Update critic 2.
        # -------------------------------------------------------------
        self.critic2_optimizer.zero_grad()

        critic2_loss.backward()

        self.critic2_optimizer.step()

        return {
            "critic1_loss": critic1_loss.item(),
            "critic2_loss": critic2_loss.item(),
            "target_q_mean": target_q.mean().item(),
            "current_q1_mean": current_q1.mean().item(),
            "current_q2_mean": current_q2.mean().item(),
        }

    def update_actor(
        self,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, float]:
        """Perform one SAC actor update."""

        observations = batch["observations"]

        # -------------------------------------------------------------
        # During the actor update, the critics act only as differentiable
        # evaluators.
        #
        # We want gradients to flow THROUGH Q(s, a) to the action,
        # but we do not want to update the critic parameters themselves.
        # -------------------------------------------------------------
        for parameter in self.critic1.parameters():
            parameter.requires_grad_(False)

        for parameter in self.critic2.parameters():
            parameter.requires_grad_(False)

        # -------------------------------------------------------------
        # Sample actions from the current stochastic policy.
        #
        # Because sample() uses rsample(), gradients can flow from the
        # resulting action back through mean/std into the actor weights.
        # -------------------------------------------------------------
        sampled_actions, log_prob, _ = self.actor.sample(
            observations
        )

        # -------------------------------------------------------------
        # Ask both critics how good the actor's current actions are.
        # -------------------------------------------------------------
        q1 = self.critic1(
            observations,
            sampled_actions,
        )

        q2 = self.critic2(
            observations,
            sampled_actions,
        )

        min_q = torch.minimum(
            q1,
            q2,
        )

        # -------------------------------------------------------------
        # SAC actor objective:
        #
        #     L_pi =
        #         E[
        #             alpha * log pi(a|s)
        #             - min(Q1(s,a), Q2(s,a))
        #          ]
        #
        # Minimizing this:
        #
        #     1. prefers actions with high Q-value
        #     2. preserves policy entropy
        # -------------------------------------------------------------
        actor_loss = (
            self.alpha.detach() * log_prob
            - min_q
        ).mean()

        # -------------------------------------------------------------
        # Gradient descent on ACTOR parameters only.
        # -------------------------------------------------------------
        self.actor_optimizer.zero_grad()

        actor_loss.backward()

        self.actor_optimizer.step()

        # -------------------------------------------------------------
        # Re-enable critic gradients for the next critic update.
        # -------------------------------------------------------------
        for parameter in self.critic1.parameters():
            parameter.requires_grad_(True)

        for parameter in self.critic2.parameters():
            parameter.requires_grad_(True)

        return {
            "actor_loss": actor_loss.item(),
            "log_prob_mean": log_prob.mean().item(),
            "actor_q_mean": min_q.mean().item(),
        }

    def update_alpha(
        self,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, float]:
        """Update the SAC entropy coefficient alpha."""

        observations = batch["observations"]

        # We only optimize log_alpha here.
        # The actor is used only to estimate the current log probability.
        with torch.no_grad():
            _, log_prob, _ = self.actor.sample(
                observations
            )

        # Standard SAC automatic entropy tuning:
        #
        #     L_alpha =
        #         -log(alpha) *
        #         (log pi(a|s) + target_entropy)
        #
        # For our four-dimensional action space:
        #
        #     target_entropy = -4
        alpha_loss = -(
            self.log_alpha
            * (
                log_prob
                + self.target_entropy
            )
        ).mean()

        self.alpha_optimizer.zero_grad()

        alpha_loss.backward()

        self.alpha_optimizer.step()

        return {
            "alpha_loss": alpha_loss.item(),
            "alpha": self.alpha.item(),
            "log_prob_mean": log_prob.mean().item(),
            "target_entropy": self.target_entropy,
        }

    def update(
        self,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, float]:
        """Perform one complete Soft Actor-Critic gradient update."""

        # -------------------------------------------------------------
        # 1. Critic update
        #
        # Learn Q1(s,a) and Q2(s,a) from the soft Bellman target.
        # -------------------------------------------------------------
        critic_metrics = self.update_critics(
            batch
        )

        # -------------------------------------------------------------
        # 2. Actor update
        #
        # Improve the stochastic policy using the critics as
        # differentiable evaluators.
        # -------------------------------------------------------------
        actor_metrics = self.update_actor(
            batch
        )

        # -------------------------------------------------------------
        # 3. Entropy-temperature update
        #
        # Adjust alpha so that policy entropy moves toward
        # the target entropy.
        # -------------------------------------------------------------
        alpha_metrics = self.update_alpha(
            batch
        )

        # -------------------------------------------------------------
        # 4. Slowly update target critics.
        #
        # target <- (1-tau) * target + tau * online
        # -------------------------------------------------------------
        polyak_update(
            target_network=self.target_critic1,
            source_network=self.critic1,
            tau=self.tau,
        )

        polyak_update(
            target_network=self.target_critic2,
            source_network=self.critic2,
            tau=self.tau,
        )

        # -------------------------------------------------------------
        # Collect useful diagnostics.
        #
        # We use distinct names because both actor and alpha updates
        # contain a log-probability statistic.
        # -------------------------------------------------------------
        return {
            "critic1_loss": critic_metrics["critic1_loss"],
            "critic2_loss": critic_metrics["critic2_loss"],
            "target_q_mean": critic_metrics["target_q_mean"],
            "current_q1_mean": critic_metrics["current_q1_mean"],
            "current_q2_mean": critic_metrics["current_q2_mean"],

            "actor_loss": actor_metrics["actor_loss"],
            "actor_log_prob_mean": actor_metrics["log_prob_mean"],
            "actor_q_mean": actor_metrics["actor_q_mean"],

            "alpha_loss": alpha_metrics["alpha_loss"],
            "alpha": alpha_metrics["alpha"],
            "alpha_log_prob_mean": alpha_metrics["log_prob_mean"],
            "target_entropy": alpha_metrics["target_entropy"],
        }