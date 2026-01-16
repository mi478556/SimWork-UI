from enum import Enum, auto


class AgentExecutionContext(Enum):


    LIVE = auto()                                         
    EVAL = auto()                                 
    PROBE = auto()                                        
    PLAYBACK = auto()                                              
