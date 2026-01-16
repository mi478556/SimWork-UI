import time
import random
import numpy as np

from env.snapshot_state import (
    EnvStateSnapshot,
    snapshot_from_runtime_dict,
    SNAPSHOT_SCHEMA_VERSION,
)

WORLD_MIN = -1.0
WORLD_MAX = 1.0
DT = 0.05


class Pod:
    def __init__(self):
        self.active = True
                                                             
        self.cooldown_until_sim = 0.0
        self.position = np.zeros(2, dtype=np.float32)
        self.spawn_position = self.position.copy()
        self.food = 0.2
        self.sequence_stage = 0

    def respawn(self, side="near"):
        self.active = True
        self.food = 0.1 + random.random() * 0.4

        if side == "near":
            self.position = np.array(
                [random.uniform(-0.9, -0.1),
                 random.uniform(-0.9, 0.9)],
                dtype=np.float32
            )
        else:
            self.position = np.array(
                [random.uniform(0.1, 0.9),
                 random.uniform(-0.9, 0.9)],
                dtype=np.float32
            )

        self.spawn_position = self.position.copy()
        self.sequence_stage = 0
                                                       
        self.cooldown_until_sim = 0.0

    @property
    def radius(self):
        return 0.05 + self.food * 0.05


class Bucket:
    def __init__(self, x, y, w=0.12, h=0.12, idx=0):
        self.rect = np.array([x, y, w, h], dtype=np.float32)
        self.idx = idx
        self.press_count = 0

    def contains(self, p):
        x, y, w, h = self.rect
        return (x <= p[0] <= x + w) and (y <= p[1] <= y + h)


class CombWall:


    def __init__(self):
        self.enabled = False
        self.open_until = 0.0
        self._get_sim_time = None

    def attach_sim_clock(self, get_sim_time):
        self._get_sim_time = get_sim_time

    @property
    def sim_time(self):
        return 0.0 if self._get_sim_time is None else self._get_sim_time()

    def is_blocking(self):
        if not self.enabled:
            return False

        # If open_until is non-positive, treat the wall as closed/blocked
        # immediately. This prevents stale open_until values from allowing
        # the wall to remain open unexpectedly.
        if self.open_until <= 0.0:
            return True

        return self.sim_time >= self.open_until

    def open_for(self, duration):
        if not self.enabled:
            return
        self.open_until = self.sim_time + duration

    def reset_timing(self):
        self.open_until = 0.0


class EnvironmentRuntime:


    DERIVED_RUNTIME_FIELDS = {
        "rooms",
        "buckets",
        "button_contact_state",
    }

    PHASE2_FOOD_THRESHOLD = 5
    PHASE3_FOOD_THRESHOLD = 12

    WALL_OPEN_DURATION = 3.0
    PHASE3_FOOD_MOVE_STEP = 0.2
    PHASE3_REACHABLE_OFFSET = 0.05

    PHASE2_TRIGGER_BUCKETS = {0, 3, 7}

    def __init__(self):
        # environments are headless; rendering is provided by EnvRenderer
        self.playback_mode = False
        self.sim_time = 0.0

                                                       
        self.agent_pos = np.array([-0.5, 0.0], dtype=np.float32)
        self.agent_vel = np.zeros(2, dtype=np.float32)
        self.agent_radius = 0.04

                                                   
        self.stomach = 0.4
        self.decay = 0.0008
        self.max_stomach = 1.6
        self.min_stomach = 0.0

                                                 
        self.phase = 1
        self.total_food_eaten_life = 0
        self.total_food_eaten_overall = 0

                                                
        self.pods = [Pod(), Pod()]
        for p in self.pods:
            p.respawn("near")

                                                           
        self.wall = CombWall()
        self.wall.attach_sim_clock(lambda: self.sim_time)

        self.buckets = self._make_buckets()
        self.bucket_side = "left"
        self.rooms = self._make_rooms_for_buckets(self.buckets, self.bucket_side)

                                                
        self.button_contact_state = {b.idx: False for b in self.buckets}

        self.sequence_order = [0, len(self.buckets) // 2, len(self.buckets) - 1]
        self.sequence_index = 0

        self._configure_phase_state()

                                                        
        self.global_step = 0

                                                                  
    def snapshot_state(self) -> EnvStateSnapshot:

        raw = {
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "step_index": self.global_step,

            "agent_pos": self.agent_pos.copy(),
            "agent_vel": self.agent_vel.copy(),

            "stomach": float(self.stomach),
            "phase": int(self.phase),

            "total_food_eaten_life": int(self.total_food_eaten_life),

            "pods": [
                {
                    "active": p.active,
                    "pos": p.position.copy(),
                    "spawn": p.spawn_position.copy(),
                    "food": float(p.food),
                    "sequence_stage": int(p.sequence_stage),
                }
                for p in self.pods
            ],

                                                               
            "wall": {
                "enabled": bool(self.wall.enabled),
                "blocking": bool(self.wall.is_blocking()),
                "open_until": float(self.wall.open_until),
            },

            "sequence_index": int(self.sequence_index),
            "bucket_side": self.bucket_side,
            "rooms": [r.copy().tolist() for r in self.rooms],
        }

        return snapshot_from_runtime_dict(raw)

                                                                  
    def _assert_invariants(self):
        expected = "left" if self.agent_pos[0] < 0 else "right"
        assert self.bucket_side == expected,\
            "Invariant violated: bucket_side must match agent sign"

        if not self.wall.enabled:
            assert not self.wall.is_blocking(),\
                "Invariant violated: wall cannot be blocking when disabled"

        assert self.sequence_index >= 0,\
            "Invariant violated: sequence_index must be >= 0"

        assert self.sequence_index <= len(self.sequence_order),\
            "Invariant violated: sequence_index exceeds sequence length"

                                                             
        for p in self.pods:
            assert WORLD_MIN <= float(p.position[0]) <= WORLD_MAX,\
                "Invariant violated: pod x-position out of world bounds"

    def apply_state(self, snap: EnvStateSnapshot, *, from_edit: bool = False):


        assert not self.playback_mode,\
            "apply_state() called during playback. This violates "\
            "the deterministic playback boundary."

                                                                        
        self.global_step = int(snap.step_index)
        self.sim_time = float(snap.step_index) * DT

                               
        self.agent_pos = snap.agent_pos.copy()
        self.agent_vel = snap.agent_vel.copy()

                         
        self.stomach = float(snap.stomach)
        self.phase = int(snap.phase)
        self.total_food_eaten_life = int(snap.total_food_eaten_life)

                                                                   
        for pod, ps in zip(self.pods, snap.pods):
            pod.active = bool(ps.active)
            pod.position = ps.pos.copy()
            pod.spawn_position = ps.spawn.copy()
            pod.food = float(ps.food)
            pod.sequence_stage = int(ps.sequence_stage)
                                                                      
            pod.cooldown_until_sim = 0.0

                                                      
        self.wall.enabled = bool(snap.wall.enabled)
        self.wall.open_until = float(snap.wall.open_until)

        # If the wall is disabled, clear any residual timing so it cannot
        # accidentally become blocking due to stale open_until values.
        if not self.wall.enabled:
            self.wall.open_until = 0.0

        _ = self.wall.is_blocking()

                                           
        self.sequence_index = int(snap.sequence_index)
        self.bucket_side = str(snap.bucket_side)

                                                                      
        # Do not recompute phase-driven runtime values here; apply_state is a
        # restoration operation and should not trigger phase logic. Update
        # wall side / rooms deterministically from provided bucket geometry.
        self._update_wall_side_to_agent()
        self.rooms = self._make_rooms_for_buckets(self.buckets, self.bucket_side)

                                                             
        self.button_contact_state = {b.idx: False for b in self.buckets}

                                    
        self._assert_invariants()

                                                                  
    def query_distance(self, a, b):
        a = np.array(a, dtype=np.float32)
        b = np.array(b, dtype=np.float32)
        return float(np.linalg.norm(a - b))

                                                                          
    def _make_buckets(self):
        buckets = []
        idx = 0
        n_buckets = 8
        w = 0.06
        h = 1.6 / n_buckets

        side = "left"
        x = -w if side == "left" else 0.0

        ys = np.linspace(-0.8, 0.8 - h, n_buckets)
        for y in ys:
            buckets.append(Bucket(x, y, w=w, h=h, idx=idx))
            idx += 1

        self.bucket_side = side
        return buckets

    def _make_rooms_for_buckets(self, buckets, side: str):
        rooms = []
        if not buckets:
            return rooms

        w = buckets[0].rect[2]
        h = buckets[0].rect[3]

        room_w = w * 2.0
        room_h = h
        room_x = -room_w if side == "left" else 0.0

        for b in buckets:
            ry = b.rect[1]
            rooms.append(np.array([room_x, ry, room_w, room_h], dtype=np.float32))

        return rooms

    def _update_wall_side_to_agent(self):
        agent_side = "left" if self.agent_pos[0] < 0.0 else "right"

        if agent_side == getattr(self, "bucket_side", None):
            return

        self.bucket_side = agent_side

        for b in self.buckets:
            w = float(b.rect[2])
            b.rect[0] = -w if agent_side == "left" else 0.0

        self.rooms = self._make_rooms_for_buckets(self.buckets, side=self.bucket_side)

    # continuous_to_screen removed; rendering is handled by env/env_renderer.py

                                                                          
    def _configure_phase_state(self):
        self.wall.enabled = (self.phase != 1)
        self.wall.reset_timing()

    def _pod_side_for_new_food(self) -> str:
        if self.phase == 1:
            return "near"
        return "far" if self.agent_pos[0] < 0.0 else "near"

    def reset(self):


        self.agent_pos = np.array([-0.5, 0.0], dtype=np.float32)
        self.agent_vel = np.zeros(2, dtype=np.float32)

        self.stomach = 0.4
        self.total_food_eaten_life = 0

        for p in self.pods:
            p.respawn(side=self._pod_side_for_new_food())

        self.wall = CombWall()
        self.wall.attach_sim_clock(lambda: self.sim_time)

        # Reset to initial phase state explicitly
        self.phase = 1
        self._configure_phase_state()

        self.buckets = self._make_buckets()
        self.rooms = self._make_rooms_for_buckets(self.buckets, side=self.bucket_side)

        self.button_contact_state = {b.idx: False for b in self.buckets}
        self.sequence_index = 0

        self.global_step = 0
        self.sim_time = 0.0

                                                                          
    def _wall_blocks_segment(self, x0, x1) -> bool:
        if not self.wall.is_blocking():
            return False
        return (x0 < 0.0 <= x1) or (x1 < 0.0 <= x0)

    def _room_blocks_segment(self, p0, p1) -> bool:
        if not self.wall.is_blocking():
            return False
        if not hasattr(self, "rooms") or not self.rooms:
            return False

        x0, y0 = float(p0[0]), float(p0[1])
        x1, y1 = float(p1[0]), float(p1[1])

        for r in self.rooms:
            rx, ry, rw, rh = r
            inside0 = (rx <= x0 <= rx + rw) and (ry <= y0 <= ry + rh)
            inside1 = (rx <= x1 <= rx + rw) and (ry <= y1 <= ry + rh)

            if (not inside0) and inside1:
                if self.bucket_side == "left":
                    if x0 >= rx:
                        return True
                else:
                    if x0 <= rx + rw:
                        return True

            if inside0 and (not inside1):
                if self.bucket_side == "left":
                    if x1 >= rx:
                        return True
                else:
                    if x1 <= rx + rw:
                        return True

            if (not inside0) and (not inside1):
                minx, maxx = min(x0, x1), max(x0, x1)
                miny, maxy = min(y0, y1), max(y0, y1)
                if not (maxx < rx or minx > rx + rw or maxy < ry or miny > ry + rh):
                    return True

        return False

    def step_physics(self, action):
        ax, ay = float(action[0]), float(action[1])
        self.agent_vel += np.array([ax, ay], dtype=np.float32) * 0.2
        self.agent_vel = np.clip(self.agent_vel, -0.3, 0.3)

        new_pos = self.agent_pos + self.agent_vel * DT
        new_pos = np.clip(new_pos, -0.98, 0.98)

        if self._wall_blocks_segment(self.agent_pos[0], new_pos[0]):
            self.agent_vel[0] = 0.0
            new_pos[0] = -0.001 if self.agent_pos[0] < 0 else 0.001

        if self._room_blocks_segment(self.agent_pos, new_pos):
            self.agent_vel[:] = 0.0
            new_pos = self.agent_pos.copy()

        self.agent_pos = new_pos

                                                                          
    def _maybe_advance_phase(self):
        if self.phase == 1 and self.total_food_eaten_life >= self.PHASE2_FOOD_THRESHOLD:
            self.phase = 2
            self._configure_phase_state()

        if self.phase == 2 and self.total_food_eaten_life >= self.PHASE3_FOOD_THRESHOLD:
            self.phase = 3
            self._configure_phase_state()
            self.sequence_index = 0

    def _on_food_eaten(self, pod: Pod):
        assert not self.playback_mode,\
            "Respawn logic must not run during playback"

        self.stomach += pod.food
        self.total_food_eaten_life += 1
        self.total_food_eaten_overall += 1

        pod.active = False
                                                                               
        cooldown_duration = random.uniform(3.0, 5.0)
        pod.cooldown_until_sim = self.sim_time + cooldown_duration

        if getattr(self, "human_mode", False):
            print(f"Stomach: {self.stomach:.3f}, total_food_life={self.total_food_eaten_life}")

        self._maybe_advance_phase()

    def _update_pods(self):
                                                      
        if self.playback_mode:
            return

        for pod in self.pods:
            if not pod.active:
                                                                   
                if self.sim_time >= pod.cooldown_until_sim and pod.cooldown_until_sim > 0.0:
                    side = self._pod_side_for_new_food()
                    pod.respawn(side=side)
                    if self.phase == 3:
                        self.sequence_index = 0
                continue

            if np.linalg.norm(self.agent_pos - pod.position) < (pod.radius + self.agent_radius):
                self._on_food_eaten(pod)

    def apply_stomach_dynamics(self):
        self.stomach -= self.decay
        if self.stomach < self.min_stomach:
            self.stomach = self.min_stomach

    def check_death(self) -> bool:
        return self.stomach > self.max_stomach

                                                                          
    def _phase2_bucket_press(self, bidx: int):
        if bidx not in self.PHASE2_TRIGGER_BUCKETS:
            return
        if not self.wall.enabled:
            return
        if not self.wall.is_blocking():
            return
        self.wall.open_for(self.WALL_OPEN_DURATION)

    def _get_active_far_pod(self):
        agent_left = self.agent_pos[0] < 0.0
        if agent_left:
            pods_active = [p for p in self.pods if p.active and p.position[0] > 0.0]
        else:
            pods_active = [p for p in self.pods if p.active and p.position[0] < 0.0]
        return pods_active[0] if pods_active else None

    def _phase3_stage_positions(self, pod):
        spawn_x = float(pod.spawn_position[0])
        pod_on_right = spawn_x > 0.0
        wall_x = 0.1 if pod_on_right else -0.1
        reach_x = -self.PHASE3_REACHABLE_OFFSET if pod_on_right else self.PHASE3_REACHABLE_OFFSET

        stage1 = spawn_x * (2.0 / 3.0) + wall_x * (1.0 / 3.0)
        stage2 = spawn_x * (1.0 / 3.0) + wall_x * (2.0 / 3.0)
        stage3 = reach_x

        return [spawn_x, stage1, stage2, stage3]

    def _phase3_bucket_press(self, bidx: int):
        pod = self._get_active_far_pod()
        if pod is None:
            return

        expected = self.sequence_order[self.sequence_index]\
            if self.sequence_index < len(self.sequence_order) else None

        if bidx != expected:
            self.sequence_index = 0
            pod.sequence_stage = 0
            pod.position[0] = float(pod.spawn_position[0])
            return

        self.sequence_index += 1
        if self.sequence_index >= len(self.sequence_order):
            self.sequence_index = 0

        pod.sequence_stage = min(pod.sequence_stage + 1, 3)

        stages = self._phase3_stage_positions(pod)
        pod.position[0] = float(stages[pod.sequence_stage])

        if pod.sequence_stage == 3:
            pod.active = True

    def _bucket_press_logic(self, bidx: int):
        if self.phase == 2:
            self._phase2_bucket_press(bidx)
        elif self.phase == 3:
            self._phase3_bucket_press(bidx)

    def _check_buckets(self):
        for b in self.buckets:
            inside = b.contains(self.agent_pos)
            was_inside = self.button_contact_state.get(b.idx, False)

            self.button_contact_state[b.idx] = inside

            if inside and not was_inside:
                b.press_count += 1
                self._bucket_press_logic(b.idx)

                                                                          
    def build_telemetry(self):
        return {
            "phase": self.phase,
            "stomach": float(self.stomach),
            "wall_enabled": self.wall.enabled,
            "wall_blocking": self.wall.is_blocking(),
            "bucket_side": self.bucket_side,
            "sequence_index": int(self.sequence_index),
        }

                                                                          
    # render/get_frame/build_observation removed; offscreen renderer provides frames

                                                                  
    def step(self, action):
        if getattr(self, "playback_mode", False):
            raise RuntimeError(
                "Playback determinism boundary violated. "
                "EnvironmentRuntime.step() called while in playback mode."
            )

                                  
        self.sim_time += DT
        self.global_step += 1

        self.step_physics(action)
        self._update_wall_side_to_agent()

        self._update_pods()
        self.apply_stomach_dynamics()
        self._check_buckets()

        death = self.check_death()

        return {
            "death": bool(death),
            "phase": self.phase,
            "stomach": float(self.stomach),
        }

                                                                  
    # get_frame and build_observation removed; observation construction moved to runners
