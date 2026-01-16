                 

from typing import Protocol


class DistanceTool(Protocol):
    def query(self, a, b) -> float:
        ...


class EnvDistanceTool:


    def __init__(self, env_runtime):
        self.env = env_runtime

    def query(self, a, b) -> float:
        return self.env.query_distance(a, b)
