# views.py               
 
from __future__ import annotations
import numpy as np

from dataset.session_types import SessionData


def training_view(session: SessionData):
    return dict(
        obs=session.frames,
        stomach=session.stomach,
        actions=session.actions,
        oracle_distance=session.oracle_distances,
    )


def oracle_distance_view(session: SessionData):
    mask = ~np.isnan(session.oracle_distances)

    return dict(
        queries=session.oracle_queries[mask],                 
        distances=session.oracle_distances[mask],        
        meta=session.meta,
    )


def playback_reconstruction_view(session: SessionData, start: int, end: int):
    s, e = int(start), int(end)

    return dict(
        meta=session.meta,

        agent_pos=session.agent_pos[s:e],
        stomach=session.stomach[s:e],
        phase=session.phases[s:e],

        food_positions=session.food_positions[s:e],

        wall_enabled=session.wall_enabled[s:e],
        wall_blocking=session.wall_blocking[s:e],

        sequence_index=session.sequence_index[s:e],

        frames=session.frames[s:e],
        actions=session.actions[s:e],
    )


def evaluation_probe_view(session: SessionData, step: int):
    i = int(step)

    return dict(
        meta=session.meta,

        step_index=i,

        agent_pos=session.agent_pos[i],
        stomach=float(session.stomach[i]),
        phase=int(session.phases[i]),

        food_positions=session.food_positions[i],

        wall_state=dict(
            enabled=bool(session.wall_enabled[i]),
            blocking=bool(session.wall_blocking[i]),
        ),

        sequence_index=int(session.sequence_index[i]),

        frame=session.frames[i],
        action=session.actions[i],

        oracle_query=session.oracle_queries[i],
        oracle_distance=session.oracle_distances[i],
    )
