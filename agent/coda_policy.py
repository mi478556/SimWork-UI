from __future__ import annotations

import os
import time
from collections import deque
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from agent.execution_context import AgentExecutionContext
from agent.coda_forward_model import CodaDynamicsConfig, CodaForwardModelComponent, CodaTokenSchema
from agent.policy_base import AgentPolicy


class CODAPolicy(AgentPolicy):
    """
    CODA: Causal Online Discovery Agent.
    Current bootstrap behavior is a wandering controller so the agent can be
    selected, run, saved, and loaded while the core design evolves.
    """

    def __init__(
        self,
        *,
        speed: float = 0.72,
        turn_every: int = 22,
        jitter: float = 0.15,
        seed: int = 101,
        enable_forward_model: bool = True,
        train_forward_model: bool = True,
        forward_buffer_size: int = 4096,
        forward_batch_size: int = 32,
        forward_warmup: int = 128,
        forward_update_every: int = 4,
        forward_lr: float = 3e-4,
        debug_dump_every: int = 200,
        debug_dir: str = "data/coda_debug",
    ):
        self.speed = float(np.clip(speed, 0.0, 1.0))
        self.turn_every = max(1, int(turn_every))
        self.jitter = float(max(0.0, jitter))
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

        self.step_count = 0
        self.direction = self._sample_unit_vec()

        self.enable_forward_model = bool(enable_forward_model)
        self.train_forward_model = bool(train_forward_model)
        self.debug_dump_every = int(max(0, debug_dump_every))
        self.debug_dir = str(debug_dir)
        self.debug_logging_enabled = self.debug_dump_every > 0

        token_schema = CodaTokenSchema(
            num_agent_slots=1,
            num_pod_slots=2,
            num_wall_slots=14,
            num_slack_slots=2,
        )
        dynamics_cfg = CodaDynamicsConfig(
            buffer_capacity=int(max(32, forward_buffer_size)),
            batch_size=int(max(1, forward_batch_size)),
            warmup_transitions=int(max(1, forward_warmup)),
            update_every=int(max(1, forward_update_every)),
            lr=float(max(1e-6, forward_lr)),
        )
        self.forward_component = CodaForwardModelComponent(
            schema=token_schema,
            dynamics_config=dynamics_cfg,
            action_dim=2,
            seed=self.seed + 31,
            training_enabled=self.train_forward_model,
            device="cpu",
        )

        self.prev_tokens: Optional[np.ndarray] = None
        self.prev_action: Optional[np.ndarray] = None
        self.prev_phase: Optional[int] = None
        self.prev_stomach: Optional[float] = None
        self.prev_agent_pos: Optional[np.ndarray] = None
        self.pending_predicted_tokens: Optional[np.ndarray] = None

        self.debug_history: deque = deque(maxlen=512)
        self.last_debug_info: Dict[str, Any] = {}
        self.latest_debug_packet: Dict[str, Any] = {}

    def _sample_unit_vec(self) -> np.ndarray:
        v = self.rng.normal(size=2).astype(np.float32)
        n = float(np.linalg.norm(v))
        if n < 1e-6:
            return np.array([1.0, 0.0], dtype=np.float32)
        return v / n

    def _normalize(self, v: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(v))
        if n < 1e-6:
            return self._sample_unit_vec()
        return (v / n).astype(np.float32)

    def _obs_to_dict(self, observation: Any) -> Dict[str, Any]:
        if isinstance(observation, dict):
            return observation
        return {}

    def _detect_reset(self, obs: Dict[str, Any]) -> bool:
        if not obs:
            return False

        phase = int(obs.get("phase", self.prev_phase if self.prev_phase is not None else 1))
        stomach = float(obs.get("stomach", self.prev_stomach if self.prev_stomach is not None else 0.4))
        pos_raw = np.asarray(
            obs.get("agent_pos", self.prev_agent_pos if self.prev_agent_pos is not None else [-0.5, 0.0]),
            dtype=np.float32,
        ).reshape(-1)
        if pos_raw.shape[0] < 2:
            pos = np.array([-0.5, 0.0], dtype=np.float32)
        else:
            pos = pos_raw[:2]

        if self.prev_phase is not None and phase < self.prev_phase:
            return True

        if self.prev_agent_pos is not None:
            jump = float(np.linalg.norm(pos - self.prev_agent_pos))
            if jump > 1.25:
                return True

        if self.prev_stomach is not None and self.prev_stomach > 1.2 and stomach < 0.55:
            if float(np.linalg.norm(pos - np.array([-0.5, 0.0], dtype=np.float32))) < 0.35:
                return True

        return False

    def _update_obs_cache(self, obs: Dict[str, Any]):
        if not obs:
            return
        self.prev_phase = int(obs.get("phase", self.prev_phase if self.prev_phase is not None else 1))
        self.prev_stomach = float(obs.get("stomach", self.prev_stomach if self.prev_stomach is not None else 0.0))
        pos_raw = np.asarray(
            obs.get("agent_pos", self.prev_agent_pos if self.prev_agent_pos is not None else [-0.5, 0.0]),
            dtype=np.float32,
        ).reshape(-1)
        if pos_raw.shape[0] >= 2:
            self.prev_agent_pos = pos_raw[:2].astype(np.float32, copy=False)

    def _dump_debug_snapshot(
        self,
        *,
        frame: Any,
        tokens: np.ndarray,
        predicted_tokens: np.ndarray,
        prediction_aligned: bool,
        debug_views: Dict[str, Any],
        summary: Dict[str, Any],
    ):
        if self.debug_dump_every <= 0:
            return

        try:
            os.makedirs(self.debug_dir, exist_ok=True)
        except Exception:
            return

        ts_ms = int(time.time() * 1000)
        stem = f"coda_step_{int(self.step_count):07d}_{ts_ms}"
        npz_path = os.path.join(self.debug_dir, f"{stem}.npz")

        frame_arr = None
        try:
            frame_arr = np.asarray(frame, dtype=np.float32) if frame is not None else None
        except Exception:
            frame_arr = None

        try:
            np.savez_compressed(
                npz_path,
                frame=frame_arr if frame_arr is not None else np.zeros((0, 0, 3), dtype=np.float32),
                tokens=np.asarray(tokens, dtype=np.float32),
                predicted_tokens=np.asarray(predicted_tokens, dtype=np.float32),
                reconstruction=np.asarray(debug_views.get("reconstruction"), dtype=np.float32),
                predicted_reconstruction=(
                    np.asarray(debug_views.get("predicted_reconstruction"), dtype=np.float32)
                    if debug_views.get("predicted_reconstruction") is not None
                    else np.zeros((0, 0, 3), dtype=np.float32)
                ),
            )
        except Exception:
            return

        json_path = os.path.join(self.debug_dir, f"{stem}.json")
        try:
            serializable = {}
            for key, value in summary.items():
                if isinstance(value, (float, int, str, bool)) or value is None:
                    serializable[key] = value
                elif isinstance(value, dict):
                    serializable[key] = {
                        str(k): (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                        for k, v in value.items()
                    }
                else:
                    serializable[key] = str(value)
            serializable["prediction_aligned"] = bool(prediction_aligned)
            import json

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(serializable, f, indent=2, sort_keys=True)
        except Exception:
            return

    def _run_forward_component(self, obs: Dict[str, Any], action: np.ndarray):
        if not self.enable_forward_model:
            return

        reset_detected = self._detect_reset(obs)
        tokens, token_diag = self.forward_component.tokenize_observation(obs)

        if reset_detected:
            self.pending_predicted_tokens = None

        if self.prev_tokens is not None and self.prev_action is not None and not reset_detected:
            self.forward_component.observe_transition(self.prev_tokens, self.prev_action, tokens)

        if reset_detected:
            self.forward_component.mark_reset()

        prediction_aligned = self.pending_predicted_tokens is not None
        aligned_predicted_tokens = (
            np.asarray(self.pending_predicted_tokens, dtype=np.float32)
            if prediction_aligned
            else None
        )

        train_metrics = self.forward_component.maybe_update()
        debug_views = self.forward_component.build_debug_views(
            frame=obs.get("frame"),
            tokens=tokens,
            predicted_next_tokens=aligned_predicted_tokens,
        )
        next_predicted_tokens = self.forward_component.predict_next_tokens(tokens, action, use_ema=True)
        predicted_tokens_for_dump = (
            aligned_predicted_tokens if aligned_predicted_tokens is not None else next_predicted_tokens
        )

        token_summary = self.forward_component.summarize_tokens(tokens)
        summary = {
            "step": int(self.step_count),
            "reset_detected": bool(reset_detected),
            "prediction_aligned": bool(prediction_aligned),
            "token_source": token_diag.get("source", "unknown"),
            "component_counts": token_diag.get("component_counts", {}),
            "wall_visible": bool(token_diag.get("wall_visible", False)),
            "active_slots": int(token_summary.get("active_slots", 0)),
            "active_per_type": token_summary.get("active_per_type", {}),
            "gate_mean": float(token_summary.get("gate_mean", 0.0)),
            "reconstruction_error": float(debug_views.get("reconstruction_error", float("nan"))),
            "predicted_reconstruction_error": (
                float(debug_views.get("predicted_reconstruction_error", float("nan")))
                if prediction_aligned
                else float("nan")
            ),
            "buffer_size": int(len(self.forward_component.buffer)),
            "total_updates": int(self.forward_component.total_updates),
            "logging_enabled": bool(self.debug_logging_enabled),
            "debug_dump_every": int(self.debug_dump_every),
            "train_updated": bool(train_metrics is not None),
        }
        metrics_src = train_metrics if train_metrics is not None else self.forward_component.last_train_metrics
        if metrics_src:
            summary["train_loss"] = float(metrics_src.get("loss", float("nan")))
            summary["train_gate_loss"] = float(metrics_src.get("gate_loss", float("nan")))
            summary["train_type_loss"] = float(metrics_src.get("type_loss", float("nan")))
            summary["train_geom_loss"] = float(metrics_src.get("geom_loss", float("nan")))

        self.last_debug_info = summary
        self.debug_history.append(summary)
        self.latest_debug_packet = {
            "step": int(self.step_count),
            "summary": dict(summary),
            "frame": np.asarray(obs.get("frame"), dtype=np.float32).copy() if obs.get("frame") is not None else None,
            "reconstruction": np.asarray(debug_views.get("reconstruction"), dtype=np.float32).copy(),
            "predicted_reconstruction": (
                np.asarray(debug_views.get("predicted_reconstruction"), dtype=np.float32).copy()
                if debug_views.get("predicted_reconstruction") is not None
                else None
            ),
            "prediction_aligned": bool(prediction_aligned),
        }

        if self.debug_logging_enabled and self.debug_dump_every > 0 and (self.step_count % self.debug_dump_every == 0):
            self._dump_debug_snapshot(
                frame=obs.get("frame"),
                tokens=tokens,
                predicted_tokens=predicted_tokens_for_dump,
                prediction_aligned=prediction_aligned,
                debug_views=debug_views,
                summary=summary,
            )

        self.prev_tokens = np.asarray(tokens, dtype=np.float32).copy()
        self.prev_action = np.asarray(action, dtype=np.float32).copy()
        self.pending_predicted_tokens = np.asarray(next_predicted_tokens, dtype=np.float32).copy()

    def act(
        self,
        observation: Any,
        tools: Dict[str, Any],
        *,
        context: AgentExecutionContext = AgentExecutionContext.LIVE,
    ) -> Tuple[list, Optional[Tuple[list, list]]]:
        self.step_count += 1

        if self.step_count % self.turn_every == 0:
            self.direction = self._sample_unit_vec()
        else:
            noise = self.rng.normal(scale=self.jitter, size=2).astype(np.float32)
            self.direction = self._normalize(self.direction + noise)

        action = (self.direction * self.speed).astype(np.float32)
        obs = self._obs_to_dict(observation)
        self._run_forward_component(obs, action)
        self._update_obs_cache(obs)
        return action, None

    def reset_episode(self):
        self.prev_tokens = None
        self.prev_action = None
        self.prev_phase = None
        self.prev_stomach = None
        self.prev_agent_pos = None
        self.pending_predicted_tokens = None
        self.latest_debug_packet = {}
        self.forward_component.mark_reset()

    def on_environment_reset(self):
        self.reset_episode()

    def get_debug_state(self) -> Dict[str, Any]:
        return {
            "last": dict(self.last_debug_info),
            "history": list(self.debug_history),
        }

    def get_latest_debug_packet(self) -> Dict[str, Any]:
        if not self.latest_debug_packet:
            return {}
        packet = {
            "step": int(self.latest_debug_packet.get("step", 0)),
            "summary": dict(self.latest_debug_packet.get("summary", {})),
            "prediction_aligned": bool(self.latest_debug_packet.get("prediction_aligned", False)),
        }
        frame = self.latest_debug_packet.get("frame")
        recon = self.latest_debug_packet.get("reconstruction")
        pred = self.latest_debug_packet.get("predicted_reconstruction")
        packet["frame"] = np.asarray(frame, dtype=np.float32).copy() if frame is not None else None
        packet["reconstruction"] = np.asarray(recon, dtype=np.float32).copy() if recon is not None else None
        packet["predicted_reconstruction"] = np.asarray(pred, dtype=np.float32).copy() if pred is not None else None
        return packet

    def set_debug_logging(self, enabled: bool, *, every: Optional[int] = None, out_dir: Optional[str] = None):
        self.debug_logging_enabled = bool(enabled)
        if every is not None:
            self.debug_dump_every = int(max(1, every))
        if out_dir is not None and str(out_dir).strip():
            self.debug_dir = str(out_dir).strip()

    def save(self, path: str):
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        payload = {
            "format": "coda_forward_v1",
            "meta": {
                "speed": float(self.speed),
                "turn_every": int(self.turn_every),
                "jitter": float(self.jitter),
                "seed": int(self.seed),
                "step_count": int(self.step_count),
                "direction": np.asarray(self.direction, dtype=np.float32),
                "rng_state": self.rng.bit_generator.state,
                "enable_forward_model": bool(self.enable_forward_model),
                "train_forward_model": bool(self.train_forward_model),
                "debug_logging_enabled": bool(self.debug_logging_enabled),
                "debug_dump_every": int(self.debug_dump_every),
                "debug_dir": str(self.debug_dir),
            },
            "forward_component": self.forward_component.state_dict(),
            "debug": {"last": dict(self.last_debug_info)},
        }
        torch.save(payload, path)

    def load(self, path: str):
        loaded_new_format = False
        try:
            try:
                payload = torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:
                payload = torch.load(path, map_location="cpu")

            if isinstance(payload, dict) and payload.get("format") == "coda_forward_v1":
                meta = payload.get("meta", {})
                self.speed = float(meta.get("speed", self.speed))
                self.turn_every = int(meta.get("turn_every", self.turn_every))
                self.jitter = float(meta.get("jitter", self.jitter))
                self.seed = int(meta.get("seed", self.seed))
                self.step_count = int(meta.get("step_count", self.step_count))

                direction = np.asarray(meta.get("direction", self.direction), dtype=np.float32).reshape(-1)
                if direction.shape[0] >= 2:
                    self.direction = self._normalize(direction[:2])
                else:
                    self.direction = self._sample_unit_vec()

                self.enable_forward_model = bool(meta.get("enable_forward_model", self.enable_forward_model))
                self.train_forward_model = bool(meta.get("train_forward_model", self.train_forward_model))
                self.debug_logging_enabled = bool(meta.get("debug_logging_enabled", self.debug_logging_enabled))
                self.debug_dump_every = int(meta.get("debug_dump_every", self.debug_dump_every))
                self.debug_dir = str(meta.get("debug_dir", self.debug_dir))

                self.rng = np.random.default_rng(self.seed)
                try:
                    rng_state = meta.get("rng_state", None)
                    if rng_state is not None:
                        self.rng.bit_generator.state = rng_state
                except Exception:
                    pass

                self.forward_component.training_enabled = self.train_forward_model
                self.forward_component.load_state_dict(payload.get("forward_component", {}))
                debug_block = payload.get("debug", {})
                self.last_debug_info = dict(debug_block.get("last", self.last_debug_info))

                self.reset_episode()
                loaded_new_format = True
        except Exception:
            loaded_new_format = False

        if loaded_new_format:
            return

        # Backward compatibility: old CODA checkpoint saved via np.savez.
        data = np.load(path, allow_pickle=False)
        self.speed = float(np.array(data.get("speed", np.array([self.speed]))).reshape(-1)[0])
        self.turn_every = int(np.array(data.get("turn_every", np.array([self.turn_every]))).reshape(-1)[0])
        self.jitter = float(np.array(data.get("jitter", np.array([self.jitter]))).reshape(-1)[0])
        self.seed = int(np.array(data.get("seed", np.array([self.seed]))).reshape(-1)[0])
        self.step_count = int(np.array(data.get("step_count", np.array([0]))).reshape(-1)[0])
        direction = np.array(data.get("direction", self.direction), dtype=np.float32).reshape(-1)
        if direction.shape[0] >= 2:
            self.direction = self._normalize(direction[:2])
        else:
            self.direction = self._sample_unit_vec()
        self.rng = np.random.default_rng(self.seed)
        self.reset_episode()
