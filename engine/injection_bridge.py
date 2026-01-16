from __future__ import annotations

import time
import numpy as np

from engine.snapshot_state import (
    EnvStateSnapshot,
    SnapshotAdvisory,

    REQUIRED_FIELDS,
    FORBIDDEN_FIELDS,
    DERIVED_FIELDS,

    SNAPSHOT_SCHEMA_VERSION,

    snapshot_from_runtime_dict,
    snapshot_to_runtime_dict,
)

WORLD_MIN = -1.0
WORLD_MAX  = 1.0


class InjectionBridge:


    def __init__(self, env_runtime):
        self.env = env_runtime

                                                          
        self.playback_lock = False


    def _clamp_vec(self, v, lo=-0.98, hi=0.98):


        return np.clip(np.array(v, dtype=np.float32), lo, hi)


    def normalize_snapshot(self, raw):


        if isinstance(raw, EnvStateSnapshot):
            snap = raw
        else:
                                             
            snap = snapshot_from_runtime_dict(raw)

                                  
        if not hasattr(snap, "schema_version"):
            snap.schema_version = SNAPSHOT_SCHEMA_VERSION

        return snap


    def validate_snapshot_preview(self, snap: EnvStateSnapshot):


        advisories: list[SnapshotAdvisory] = []

                                          
        missing = [f for f in REQUIRED_FIELDS if not hasattr(snap, f)]
        if missing:
            advisories.append(SnapshotAdvisory(
                "reject",
                f"Snapshot missing required fields: {missing}"
            ))
            return snap, advisories

                                          
        for name in FORBIDDEN_FIELDS | DERIVED_FIELDS:
            if hasattr(snap, name):
                advisories.append(SnapshotAdvisory(
                    "reject",
                    f"Snapshot contains forbidden runtime field '{name}'. "
                    "Derived runtime values must be recomputed "
                    "by EnvironmentRuntime.apply_state()."
                ))
                return snap, advisories

                                          
        clamped = self._clamp_vec(snap.agent_pos)
        if not np.allclose(clamped, snap.agent_pos):
            advisories.append(SnapshotAdvisory(
                "safe",
                "Agent position clamped to world bounds"
            ))
            snap.agent_pos = clamped

                              
        if snap.stomach < 0 or snap.stomach > 10:
            advisories.append(SnapshotAdvisory(
                "safe",
                "Stomach value clamped to legal range [0, 10]"
            ))
            snap.stomach = max(0.0, min(float(snap.stomach), 10.0))

                                      
        expected_side = "left" if snap.agent_pos[0] < 0 else "right"
        if snap.bucket_side != expected_side:
            advisories.append(SnapshotAdvisory(
                "risky",
                "Bucket side adjusted to match agent side"
            ))
            snap.bucket_side = expected_side

                             
        for p in snap.pods:
            new_pos = self._clamp_vec(p.pos)
            if not np.allclose(new_pos, p.pos):
                advisories.append(SnapshotAdvisory(
                    "safe",
                    "Pod position clamped to world bounds"
                ))
                p.pos = new_pos

                                                               
        for p in snap.pods:
            if np.linalg.norm(snap.agent_pos - p.pos) < 0.05:
                advisories.append(SnapshotAdvisory(
                    "risky",
                    "Agent placed inside pod — collision will resolve on injection"
                ))

                           
        if hasattr(snap.wall, "open_until") and snap.wall.open_until < 0:
            advisories.append(SnapshotAdvisory(
                "safe",
                "Wall open_until clamped >= 0"
            ))
            snap.wall.open_until = 0.0

                                      
        if not snap.wall.enabled and snap.wall.blocking:
            advisories.append(SnapshotAdvisory(
                "risky",
                "Wall blocking contradicted enabled flag — corrected"
            ))
            snap.wall.blocking = False

                                       
        if snap.sequence_index < 0:
            advisories.append(SnapshotAdvisory(
                "reject",
                "Negative sequence index is invalid"
            ))
            return snap, advisories

        return snap, advisories


    def assert_not_in_playback(self):


        if self.playback_lock:
            raise RuntimeError(
                "Deterministic playback boundary violated — "
                "state mutation attempted during playback mode"
            )


    def apply_snapshot(self, raw, *, from_edit: bool = False):


        self.assert_not_in_playback()

        snap = self.normalize_snapshot(raw)

        snap, advisories = self.validate_snapshot_preview(snap)

                               
        if any(a.level == "reject" for a in advisories):
            raise ValueError(
                "Snapshot injection rejected:\n" +
                "\n".join(f"- [{a.level}] {a.message}" for a in advisories)
            )

                                               
        # apply the EnvStateSnapshot directly to the runtime
        # pass through edit-origin flag so runtime can treat god-mode edits specially
        self.env.apply_state(snap, from_edit=from_edit)

                                   
        try:
            self.env._update_wall_side_to_agent()
        except Exception:
            pass

        return snap, advisories
