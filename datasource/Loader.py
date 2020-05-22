from abc import ABC, abstractmethod
class Loader(ABC): 
    def __init__(self, path):
        self.path=path
    
    @abstractmethod #decorator for defining abstract classes   
    def load(self):
        pass