from abc import abstractmethod

class BaseAgent():
    actionTrigger = False
    def __init__(self,Inventory:int=0,ServiceTime:int=1,\
                 stockoutcost=100000,inventorycost=[1000,5,1000],penalty=10000):
        self.Inventory = Inventory
        self.penalty = penalty
        self.stockoutcost = stockoutcost
        self.inventorycost = inventorycost

    @abstractmethod
    def _updateInventory(self):
        raise NotImplemented
    
    @abstractmethod
    def _updateDemand(self):
        raise NotImplemented
    
    @abstractmethod
    def _getReward(self):
        raise NotImplemented
    
    @abstractmethod
    def nextAction(self):
        raise NotImplemented
