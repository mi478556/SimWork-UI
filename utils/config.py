                 

from dataclasses import dataclass


@dataclass(frozen=True)
class WorldConfig:
    world_min: float = -1.0
    world_max: float = 1.0
    screen_size: int = 256
    dt: float = 0.05


@dataclass(frozen=True)
class PhasingConfig:
    phase2_food_threshold: int = 5
    phase3_food_threshold: int = 12


@dataclass(frozen=True)
class WallConfig:
    wall_open_duration: float = 3.0
    phase2_trigger_buckets = (0, 3, 7)


@dataclass(frozen=True)
class Phase3Config:
    move_step: float = 0.2
    reachable_offset: float = 0.05
    seq_buttons = (0, 4, 7)                            


@dataclass(frozen=True)
class RecordingConfig:
    default_session_fps: int = 20
    max_clip_length: int = 10_000
    allow_human_recording: bool = False


WORLD = WorldConfig()
PHASES = PhasingConfig()
WALL = WallConfig()
PHASE3 = Phase3Config()
REC = RecordingConfig()
