from abc import ABC, abstractmethod

class BaseGrader(ABC):
    @abstractmethod
    def grade(self, episode_state) -> float:
        pass
