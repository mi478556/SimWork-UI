# policy_base.py

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

from agent.execution_context import AgentExecutionContext


@dataclass
class Observation:


    frame: Any                                             
    stomach: float
    oracle_distance: Optional[float] = None

                                                 
    prev_action: Optional[list] = None
    query_points: Optional[Any] = None


class AgentPolicy:


    def act(
        self,
        observation: Observation,
        tools: Dict[str, Any],
        *,
        context: AgentExecutionContext = AgentExecutionContext.LIVE,
    ) -> Tuple[list, Optional[Tuple[list, list]]]:


        raise NotImplementedError
