# event_bus.py

from collections import defaultdict
from typing import Callable, Dict, List


class EventBus:


    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)

    def subscribe(self, event_type: str, handler: Callable):
        self._subscribers[event_type].append(handler)

    def unsubscribe(self, event_type: str, handler: Callable):
        if handler in self._subscribers[event_type]:
            self._subscribers[event_type].remove(handler)

    def publish(self, event_type: str, payload=None):


        listeners = self._subscribers.get(event_type, None)

        if not listeners:
                                                                       
                                                    
            return

        for fn in list(listeners):
            fn(payload)
