import numpy as np
import torch


class ReplayBuffer:
    """Fixed-size replay buffer for off-policy reinforcement learning."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        capacity: int,
        device: str = "cpu",
        seed: int | None = None,
    ):
        # -------------------------------------------------------------
        # Basic configuration
        # -------------------------------------------------------------
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.capacity = capacity
        self.device = torch.device(device)

        # Independent random-number generator for reproducible sampling.
        self.rng = np.random.default_rng(seed)

        # -------------------------------------------------------------
        # Pre-allocate memory.
        #
        # We know all array dimensions in advance, so allocating one
        # large block is much more efficient than continuously appending
        # Python objects to a list.
        # -------------------------------------------------------------
        self.observations = np.zeros(
            (capacity, observation_dim),
            dtype=np.float32,
        )

        self.actions = np.zeros(
            (capacity, action_dim),
            dtype=np.float32,
        )

        self.rewards = np.zeros(
            (capacity, 1),
            dtype=np.float32,
        )

        self.next_observations = np.zeros(
            (capacity, observation_dim),
            dtype=np.float32,
        )

        # 1.0 = true terminal state
        # 0.0 = transition can bootstrap
        self.terminated = np.zeros(
            (capacity, 1),
            dtype=np.float32,
        )

        # -------------------------------------------------------------
        # Circular-buffer state
        #
        # position:
        #   index where the next transition will be written
        #
        # size:
        #   number of valid transitions currently stored
        # -------------------------------------------------------------
        self.position = 0
        self.size = 0

    def __len__(self) -> int:
        """Return the number of valid transitions currently stored."""
        return self.size

    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_observation: np.ndarray,
        terminated: bool,
    ) -> None:
        """Store one transition in the replay buffer."""

        observation = np.asarray(
            observation,
            dtype=np.float32,
        )

        action = np.asarray(
            action,
            dtype=np.float32,
        )

        next_observation = np.asarray(
            next_observation,
            dtype=np.float32,
        )

        # -------------------------------------------------------------
        # Basic shape checks.
        #
        # These help catch environment/agent interface bugs early.
        # -------------------------------------------------------------
        if observation.shape != (self.observation_dim,):
            raise ValueError(
                f"Expected observation shape "
                f"({self.observation_dim},), "
                f"got {observation.shape}."
            )

        if action.shape != (self.action_dim,):
            raise ValueError(
                f"Expected action shape "
                f"({self.action_dim},), "
                f"got {action.shape}."
            )

        if next_observation.shape != (self.observation_dim,):
            raise ValueError(
                f"Expected next_observation shape "
                f"({self.observation_dim},), "
                f"got {next_observation.shape}."
            )

        # -------------------------------------------------------------
        # Write the transition at the current circular-buffer position.
        # -------------------------------------------------------------
        self.observations[self.position] = observation
        self.actions[self.position] = action
        self.rewards[self.position, 0] = reward
        self.next_observations[self.position] = next_observation
        self.terminated[self.position, 0] = float(terminated)

        # -------------------------------------------------------------
        # Advance the write pointer.
        #
        # Once we reach capacity, modulo wraps back to index 0.
        # -------------------------------------------------------------
        self.position = (
            self.position + 1
        ) % self.capacity

        # Size grows until the buffer is full.
        self.size = min(
            self.size + 1,
            self.capacity,
        )

    def sample(
        self,
        batch_size: int,
    ) -> dict[str, torch.Tensor]:
        """Uniformly sample a mini-batch and return PyTorch tensors."""

        if self.size < batch_size:
            raise ValueError(
                f"Cannot sample batch_size={batch_size} "
                f"from buffer containing only {self.size} transitions."
            )

        # -------------------------------------------------------------
        # Uniform random sampling from all currently valid transitions.
        # -------------------------------------------------------------
        indices = self.rng.integers(
            low=0,
            high=self.size,
            size=batch_size,
        )

        # -------------------------------------------------------------
        # Convert NumPy data to PyTorch tensors.
        #
        # From this point onward these tensors can be sent directly
        # through our Actor and Critic neural networks.
        # -------------------------------------------------------------
        batch = {
            "observations": torch.as_tensor(
                self.observations[indices],
                device=self.device,
            ),
            "actions": torch.as_tensor(
                self.actions[indices],
                device=self.device,
            ),
            "rewards": torch.as_tensor(
                self.rewards[indices],
                device=self.device,
            ),
            "next_observations": torch.as_tensor(
                self.next_observations[indices],
                device=self.device,
            ),
            "terminated": torch.as_tensor(
                self.terminated[indices],
                device=self.device,
            ),
        }

        return batch