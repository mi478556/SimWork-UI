from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn

from agent.execution_context import AgentExecutionContext
from agent.policy_base import AgentPolicy


class _ActorCriticNet(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, init_std: float):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )
        self.log_std = nn.Parameter(torch.log(torch.full((action_dim,), float(max(1e-4, init_std)))))

    def policy_mean(self, obs: torch.Tensor) -> torch.Tensor:
        # Bound means to action-like range while keeping stochastic policy from Normal.
        return torch.tanh(self.actor(obs))

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)


class RLActorCriticFoodPolicy(AgentPolicy):
    """
    PPO-style actor-critic policy for food-pod collection.
    - On-policy rollouts
    - GAE(lambda) advantages
    - PPO clipped surrogate updates
    - Entropy regularization
    Learning data is gathered only during phase 1.
    """

    def __init__(
        self,
        *,
        gamma: float = 0.98,
        gae_lambda: float = 0.95,
        actor_lr: float = 3e-4,
        critic_lr: float = 1e-3,
        clip_ratio: float = 0.2,
        ppo_epochs: int = 6,
        minibatch_size: int = 128,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        action_std: float = 0.35,
        min_std: float = 0.05,
        max_std: float = 0.8,
        max_grad_norm: float = 0.7,
        seed: int = 17,
    ):
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.actor_lr = float(actor_lr)
        self.critic_lr = float(critic_lr)
        self.clip_ratio = float(max(0.01, clip_ratio))
        self.ppo_epochs = int(max(1, ppo_epochs))
        self.minibatch_size = int(max(8, minibatch_size))
        self.value_coef = float(max(0.0, value_coef))
        self.entropy_coef = float(max(0.0, entropy_coef))
        self.min_std = float(max(1e-4, min_std))
        self.max_std = float(max(self.min_std, max_std))
        self.max_grad_norm = float(max(1e-6, max_grad_norm))

        self.feature_dim = 9
        self.action_dim = 2
        self.device = torch.device("cpu")

        self.model = _ActorCriticNet(self.feature_dim, self.action_dim, init_std=action_std).to(self.device)
        self.actor_optimizer = torch.optim.Adam(
            list(self.model.actor.parameters()) + [self.model.log_std],
            lr=self.actor_lr,
        )
        self.critic_optimizer = torch.optim.Adam(self.model.critic.parameters(), lr=self.critic_lr)

        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        # One-step delayed reward bookkeeping
        self.prev_features: Optional[np.ndarray] = None
        self.prev_raw_action: Optional[np.ndarray] = None
        self.prev_logp: Optional[float] = None
        self.prev_value: Optional[float] = None
        self.prev_stomach: Optional[float] = None
        self.prev_phase: Optional[int] = None
        self.prev_nearest_dist: Optional[float] = None

        # On-policy rollout buffer (phase 1 only)
        self.rollout: List[Dict[str, Any]] = []

        # Dashboard/public state
        self.learning_enabled = True
        self.total_updates = 0
        self.last_reward = 0.0
        self.episode_reward = 0.0
        self.best_episode_reward = float("-inf")
        self.episodes_seen = 0

    def _nearest_active_pod_delta(self, obs: Dict[str, Any]) -> Tuple[np.ndarray, float]:
        agent_pos = np.array(obs.get("agent_pos", [0.0, 0.0]), dtype=np.float32)
        pods = obs.get("pods", []) or []

        best_delta = np.zeros(2, dtype=np.float32)
        best_dist = 2.0
        found = False

        for p in pods:
            if not p.get("active", False):
                continue
            pod_pos = np.array(p.get("pos", [0.0, 0.0]), dtype=np.float32)
            delta = pod_pos - agent_pos
            dist = float(np.linalg.norm(delta))
            if (not found) or dist < best_dist:
                found = True
                best_dist = dist
                best_delta = delta

        if not found:
            return np.zeros(2, dtype=np.float32), 2.0
        return best_delta, best_dist

    def _build_features(self, obs: Dict[str, Any]) -> np.ndarray:
        stomach = float(obs.get("stomach", 0.0))
        phase = int(obs.get("phase", 1))
        agent_vel = np.array(obs.get("agent_vel", [0.0, 0.0]), dtype=np.float32)
        nearest_delta, nearest_dist = self._nearest_active_pod_delta(obs)

        if nearest_dist > 1e-6:
            dir_to_food = nearest_delta / nearest_dist
        else:
            dir_to_food = np.zeros(2, dtype=np.float32)

        return np.array(
            [
                1.0,
                np.clip(stomach / 1.6, 0.0, 2.0),
                1.0 if phase == 1 else 0.0,
                np.clip(dir_to_food[0], -1.0, 1.0),
                np.clip(dir_to_food[1], -1.0, 1.0),
                np.clip(nearest_dist / 2.0, 0.0, 1.0),
                np.clip(float(agent_vel[0]), -1.0, 1.0),
                np.clip(float(agent_vel[1]), -1.0, 1.0),
                np.clip(np.linalg.norm(agent_vel), 0.0, 1.0),
            ],
            dtype=np.float32,
        )

    def _compute_reward(self, obs: Dict[str, Any]) -> float:
        stomach = float(obs.get("stomach", 0.0))
        _, nearest_dist = self._nearest_active_pod_delta(obs)

        if self.prev_stomach is None:
            self.prev_stomach = stomach
        if self.prev_nearest_dist is None:
            self.prev_nearest_dist = nearest_dist

        stomach_delta = stomach - float(self.prev_stomach)
        dist_progress = float(self.prev_nearest_dist) - nearest_dist

        # Dense shaping for faster learning in this environment.
        reward = (10.0 * stomach_delta) + (0.4 * dist_progress) - 0.001
        return float(reward)

    def _current_std(self) -> torch.Tensor:
        return torch.exp(self.model.log_std).clamp(self.min_std, self.max_std)

    def _policy_forward(self, obs_t: torch.Tensor):
        mean = self.model.policy_mean(obs_t)
        value = self.model.value(obs_t)
        std = self._current_std().expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        return dist, value

    def _squashed_log_prob(self, dist, raw_action: torch.Tensor) -> torch.Tensor:
        action = torch.tanh(raw_action)
        logp = dist.log_prob(raw_action).sum(dim=-1)
        correction = torch.log(1.0 - action.pow(2) + 1e-6).sum(dim=-1)
        return logp - correction

    def _append_prev_transition(self, reward: float, done: bool):
        if self.prev_features is None or self.prev_raw_action is None:
            return
        if self.prev_logp is None or self.prev_value is None:
            return
        if self.prev_phase is not None and self.prev_phase >= 2:
            return

        self.rollout.append(
            {
                "obs": self.prev_features.copy(),
                "raw_action": self.prev_raw_action.copy(),
                "old_logp": float(self.prev_logp),
                "old_value": float(self.prev_value),
                "reward": float(reward),
                "done": bool(done),
            }
        )

    def _ppo_update(self):
        if not self.rollout:
            return

        self.model.train()

        obs_np = np.stack([t["obs"] for t in self.rollout], axis=0).astype(np.float32, copy=False)
        raw_actions_np = np.stack([t["raw_action"] for t in self.rollout], axis=0).astype(np.float32, copy=False)
        old_logp_np = np.array([t["old_logp"] for t in self.rollout], dtype=np.float32)
        old_values_np = np.array([t["old_value"] for t in self.rollout], dtype=np.float32)
        rewards_np = np.array([t["reward"] for t in self.rollout], dtype=np.float32)
        dones_np = np.array([t["done"] for t in self.rollout], dtype=np.float32)

        obs = torch.from_numpy(obs_np).to(self.device)
        raw_actions = torch.from_numpy(raw_actions_np).to(self.device)
        old_logp = torch.from_numpy(old_logp_np).to(self.device)
        old_values = torch.from_numpy(old_values_np).to(self.device)
        rewards = torch.from_numpy(rewards_np).to(self.device)
        dones = torch.from_numpy(dones_np).to(self.device)

        n = rewards.shape[0]

        # GAE(lambda)
        advantages = torch.zeros(n, dtype=torch.float32, device=self.device)
        returns = torch.zeros(n, dtype=torch.float32, device=self.device)
        gae = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        next_value = torch.tensor(0.0, dtype=torch.float32, device=self.device)

        for t in range(n - 1, -1, -1):
            non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * non_terminal - old_values[t]
            gae = delta + self.gamma * self.gae_lambda * non_terminal * gae
            advantages[t] = gae
            returns[t] = advantages[t] + old_values[t]
            next_value = old_values[t]

        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        batch_size = min(self.minibatch_size, int(n))

        for _ in range(self.ppo_epochs):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, batch_size):
                mb = perm[start : start + batch_size]

                dist, values = self._policy_forward(obs[mb])
                new_logp = self._squashed_log_prob(dist, raw_actions[mb])
                ratio = torch.exp((new_logp - old_logp[mb]).clamp(-20.0, 20.0))

                adv_mb = advantages[mb]
                surr1 = ratio * adv_mb
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv_mb
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = 0.5 * (returns[mb] - values).pow(2).mean()

                entropy = dist.entropy().sum(dim=-1).mean()
                actor_loss = policy_loss - self.entropy_coef * entropy

                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.actor.parameters()) + [self.model.log_std], self.max_grad_norm
                )
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad(set_to_none=True)
                (self.value_coef * value_loss).backward()
                torch.nn.utils.clip_grad_norm_(self.model.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                self.total_updates += 1

    def apply_terminal_reward(self, reward: float):
        r = float(reward)
        self.last_reward = r
        self.episode_reward += r
        self._append_prev_transition(r, done=True)

        # Terminal transition consumed.
        self.prev_features = None
        self.prev_raw_action = None
        self.prev_logp = None
        self.prev_value = None

    def reset_episode(self):
        # Close any dangling transition with terminal zero reward.
        if self.prev_features is not None and self.prev_phase is not None and self.prev_phase < 2:
            self._append_prev_transition(0.0, done=True)

        self._ppo_update()

        if self.episode_reward > self.best_episode_reward:
            self.best_episode_reward = float(self.episode_reward)
        self.episodes_seen += 1

        self.episode_reward = 0.0
        self.rollout = []

        self.prev_features = None
        self.prev_raw_action = None
        self.prev_logp = None
        self.prev_value = None
        self.prev_stomach = None
        self.prev_phase = None
        self.prev_nearest_dist = None

        self.learning_enabled = True

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "model": self.model.state_dict(),
            "actor_opt": self.actor_optimizer.state_dict(),
            "critic_opt": self.critic_optimizer.state_dict(),
            "meta": {
                "gamma": self.gamma,
                "gae_lambda": self.gae_lambda,
                "actor_lr": self.actor_lr,
                "critic_lr": self.critic_lr,
                "clip_ratio": self.clip_ratio,
                "ppo_epochs": self.ppo_epochs,
                "minibatch_size": self.minibatch_size,
                "value_coef": self.value_coef,
                "entropy_coef": self.entropy_coef,
                "min_std": self.min_std,
                "max_std": self.max_std,
                "max_grad_norm": self.max_grad_norm,
                "total_updates": self.total_updates,
                "last_reward": self.last_reward,
                "episode_reward": self.episode_reward,
                "best_episode_reward": self.best_episode_reward,
                "episodes_seen": self.episodes_seen,
            },
        }
        torch.save(payload, path)

    def load(self, path: str):
        try:
            payload = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location=self.device)
        self.model.load_state_dict(payload["model"])

        try:
            self.actor_optimizer.load_state_dict(payload["actor_opt"])
            self.critic_optimizer.load_state_dict(payload["critic_opt"])
        except Exception:
            # Optimizer state is nice-to-have; weights are authoritative.
            pass

        meta = payload.get("meta", {})
        self.total_updates = int(meta.get("total_updates", 0))
        self.last_reward = float(meta.get("last_reward", 0.0))
        self.episode_reward = float(meta.get("episode_reward", 0.0))
        self.best_episode_reward = float(meta.get("best_episode_reward", self.best_episode_reward))
        self.episodes_seen = int(meta.get("episodes_seen", 0))

        # Runtime buffers are always reset on load.
        self.rollout = []
        self.prev_features = None
        self.prev_raw_action = None
        self.prev_logp = None
        self.prev_value = None
        self.prev_stomach = None
        self.prev_phase = None
        self.prev_nearest_dist = None
        self.learning_enabled = True

    def act(
        self,
        observation: Any,
        tools: Dict[str, Any],
        *,
        context: AgentExecutionContext = AgentExecutionContext.LIVE,
    ) -> Tuple[list, Optional[Tuple[list, list]]]:
        obs = observation if isinstance(observation, dict) else {}
        phase = int(obs.get("phase", 1))

        # Reward for prior action becomes available on current state.
        if self.prev_features is not None:
            reward = self._compute_reward(obs)
            self.last_reward = float(reward)
            self.episode_reward += float(reward)
            self._append_prev_transition(reward, done=False)

        if phase >= 2:
            self.learning_enabled = False

        features = self._build_features(obs)
        obs_t = torch.from_numpy(features).to(self.device).unsqueeze(0)

        self.model.eval()
        with torch.no_grad():
            dist, value = self._policy_forward(obs_t)
            if self.learning_enabled and phase == 1:
                raw_action = dist.sample()
            else:
                # Deterministic policy after freeze
                raw_action = dist.mean

            action = torch.tanh(raw_action)
            logp = self._squashed_log_prob(dist, raw_action)

        action_np = action.squeeze(0).cpu().numpy().astype(np.float32)
        raw_np = raw_action.squeeze(0).cpu().numpy().astype(np.float32)

        # Cache transition head for next-step reward assignment.
        self.prev_features = features
        self.prev_raw_action = raw_np
        self.prev_logp = float(logp.item())
        self.prev_value = float(value.squeeze(0).item())
        self.prev_stomach = float(obs.get("stomach", 0.0))
        _, self.prev_nearest_dist = self._nearest_active_pod_delta(obs)
        self.prev_phase = phase

        return action_np, None
