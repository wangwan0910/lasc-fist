# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 19:02:22 2023

@author: ww1a23
"""

import pandas as pd
import numpy as np
import sys
import os
import gym

current_dir = os.path.dirname(__file__)
sys.path.append(current_dir)

from utilss.utils import Queue, getdata, CustomSpace
from gymnasium import Env, spaces
import random

from common import BaseAgent


# pre_dir = os.path.dirname(current_dir)
# sys.path.append(pre_dir)
# print(pre_dir, sys.path)
#
# from env_list import ENV_LIST
# print("------------------------", ENV_LIST)


random_seed = 2023
np.random.seed(random_seed)
random.seed()

lives = 0


class Customer():
    def __init__(self, muDemand, stdDemand):
        ''' 
        Initializes an instance of the Customer class.
        Parameters:
        - muDemand: The mean demand value (default: 2).
        - stdDemand: The standard deviation of demand (default: 0).
       
        '''

        self.muDemand = muDemand
        self.stdDemand = stdDemand
        self.index = 0

    def _getDemand(self):
        '''
        Internal method: Generates demand value following a normal distribution.
        Returns:
        - demand: The demand value following a normal distribution (rounded down to the nearest integer).
      
        '''

        demand = max(0, np.random.normal(self.muDemand, self.stdDemand))
        demand = np.floor(demand)
        # print('demand1111',demand)
        return demand

    def getOrder(self) -> int:
        '''
        
        Retrieves the customer's demand order.
        Returns:
        - order: The customer's demand order (integer).
     
        '''
        return self._getDemand()


class Retailer(BaseAgent):
    def __init__(self, customer: Customer, Inventory: int = 10, ServiceTime: int = 0, stockout: int = 0,
                 Demand: int = 0):
        '''
        Initializes an instance of the Retailer class.
        Parameters:
        - customer: An instance of the Customer class representing the retailer's customer.
        - Inventory: The initial inventory level of the retailer (default: 10).
        - ServiceTime: The length of the service queue (default: 0).
        - stockout: The number of stockouts (default: 0).
        - Demand: The demand value (default: 0).
        '''
        super(Retailer, self).__init__()
        self.ServiceQueue = Queue(ServiceTime)
        # Initialize the service queue with zeros
        for i in range(ServiceTime):
            self.ServiceQueue.enqueue(0)
        self.Inventory = Inventory
        self.order = 10
        self.customer = customer
        self.reorderPoint = 5
        self.Sum_stockout = 0
        self.stockoutcost = 10000
        self.stockout = stockout
        self.Demand = Demand
        self.preOrder = 0

    def _updatServiceQueue(self):
        '''
        Internal method: Updates the service queue by adding a new demand value from the customer.
        '''
        self.Demand = self.customer.getOrder()
        # print(self.Demand)
        self.ServiceQueue.enqueue(self.Demand)

    def updatereorderPoint(self, reorderPoint):
        '''
        Updates the reorder point for the retailer.
        Parameters:
        - reorderPoint: The new reorder point value.
      
        '''
        self.reorderPoint = reorderPoint

    def _isOrder(self, other: BaseAgent) -> bool:
        '''
        Internal method: Checks if the retailer should place an order based on inventory, service queue, and other conditions.
        Parameters:
        - other: Another agent (not used in the code).
        Returns:
        - True if the retailer should place an order, False otherwise.
        '''
        # print( '1',(self.Inventory <= self.reorderPoint) )
        # print(('2',other.ServiceQueue.getQueue().copy() == [0,0,0]) )
        # print('3', other.WOrderlist[-1]==0)
        return (self.Inventory <= self.reorderPoint) and (other.ServiceQueue.getQueue().copy() == [0, 0, 0]) and (
                    other.WOrderlist[-1] == 0)

    def sandactionTrigger(self, other: BaseAgent, reorderPoint):
        '''
        Checks if the retailer should trigger an action (place an order) based on the conditions and updates the reorder point.
        Parameters:
        - other: Another agent (not used in the code).
        - reorderPoint: The reorder point value.
        
        '''
        self.reorderPoint = reorderPoint

        if self._isOrder(other):
            BaseAgent.actionTrigger = True

    # def updateOrder(self,order):
    '''
 
        Updates the order value for the retailer.
        Parameters:
        - order: The new order value.
   
    '''

    #     self.order = order

    def getOrderRequest(self) -> int:
        '''
        Retrieves the order request for the retailer.
        Returns:
        - The order value if an action trigger is True, otherwise returns 0.
        '''
        if BaseAgent.actionTrigger:
            return self.order
        else:

            return 0

    def updateInventory(self, preOrder):
        '''
        Updates the retailer's inventory based on the demand and previous order.
        Parameters:
        - preOrder: The previous order value.
     
        '''

        self.Demand = self.ServiceQueue.dequeue()
        self.preOrder = preOrder

        self.Inventory += self.preOrder
        # self.stockout
        if self.Demand <= self.Inventory:
            self.Inventory -= self.Demand

        else:
            # print('11111177777777777777',self.stockout)
            # print(self.Demand,self.Inventory)
            self.stockout += self.Demand - self.Inventory

            # print('33333333333',self.stockout)
            global lives
            lives += 1
            # self.Sum_stockout += self.stockout
            # 

            # self.Demand = self.Inventory
            # self.Creal_order = self.Inventory
            self.Inventory = 0

        # print(' 7777777777777777777' ,self.stockout)
        # print('self.Creal_order ',self.Creal_order )

        # print(' 7' ,self.stockout)

        # if BaseAgent.actionTrigger:
        #         self.stockout = 0
        if self.Inventory > 30:
            self.Inventory = 30
        if self.Inventory < 0:
            self.Inventory = 0

    def getReward(self):
        '''
     
        Calculates the reward for the retailer based on inventory and stockout costs.
        Returns:
        - The reward value.
    
        '''

        reward = 0

        reward -= self.Inventory * self.inventorycost[0]
        reward -= self.stockout * self.stockoutcost
        # print(' 9self.reward' ,reward,self.stockout)
        # print('rr11',reward ,self.Inventory)

        # print('rr',reward ,self.stockout)

        return reward

    def _getObs(self):
        '''
        Internal method: Retrieves the observation state of the retailer.
        Returns:
        - The inventory and a copy of the service queue.
        '''
        return self.Inventory, self.ServiceQueue.getQueue().copy()

    def nextAction(self, reorderPoint):
        '''
 
        Determines the next action for the retailer based on order, other agent, and reorder point.
        Parameters:
        - order: The order value.
        - reorderPoint: The reorder point value.
        Returns:
        - The observation state of the retailer.
        
        '''
        self.updatereorderPoint(reorderPoint)
        # self.sandactionTrigger(other,reorderPoint)
        self._updatServiceQueue()

        return self._getObs()


class Warehouse(BaseAgent):
    def __init__(self, retailer: Retailer, Inventory: int = 0, ServiceTime: int = 3, Demand: int = 0,
                 Rreal_order: int = 0):
        '''
        Initializes an instance of the Warehouse class.
        Parameters:
        - retailer: An instance of the Retailer class representing the warehouse's retailer.
        - Inventory: The initial inventory level of the warehouse (default: 0).
        - ServiceTime: The length of the service queue (default: 3).
        - Demand: The demand value (default: 0).
        - Rreal_order: The retailer real order value (default: 0).
        '''
        super(Warehouse, self).__init__()
        self.ServiceQueue = Queue(ServiceTime)
        # Initialize the service queue with zeros
        for i in range(ServiceTime):
            self.ServiceQueue.enqueue(0)
        self.Inventory = Inventory
        self.preOrder = 0
        self.order = 0
        self.retailer = retailer
        self.WOrderlist = []
        self.Rreal_order = Rreal_order
        self.Demand = Demand

    def _updatServiceQueue(self):
        '''
        Internal method: Updates the service queue by adding a new demand value from the retailer.    
        '''
        self.Demand = self.retailer.getOrderRequest()
        self.ServiceQueue.enqueue(self.Demand)

    def updateOrder(self, order):
        '''
        Updates the order value for the warehouse.
        Parameters:
        - order: The new order value.
        '''
        self.order = order

    def getOrderRequest(self) -> int:
        '''
        Retrieves the order request for the warehouse.
        Returns:
        - The order value if an action trigger is True, otherwise returns 0.
        
        '''
        if BaseAgent.actionTrigger:

            return self.order
        else:

            return 0

    def updateInventory(self, preOrder):
        '''
 
        Updates the warehouse's inventory based on the previous order.
        Parameters:
        - preOrder: The previous order value.
      
        '''
        self.preOrder = preOrder
        self.Demand = self.ServiceQueue.dequeue()
        self.WOrderlist = np.append(self.WOrderlist, [self.Demand])

        self.Inventory += self.preOrder

        if self.Demand <= self.Inventory:
            self.Inventory -= self.Demand

            self.Rreal_order = self.Demand
            self.retailer.updateInventory(self.Demand)
        else:
            # print("Insufficient Inventory.")
            self.Demand = self.Inventory
            self.Rreal_order = self.Inventory

            self.retailer.updateInventory(self.Demand)

            self.Inventory = 0
        # print('self.Rreal_order ',self.Rreal_order )
        # print('**always true',self.Rreal_order == self.retailer.preOrder)
        # self.Inventory -=self.retailer.preOrder

        if self.Inventory > 30:
            self.Inventory = 30
        if self.Inventory < 0:
            self.Inventory = 0
        # self.retailer.sandactionTrigger(self,5):#####???

    def getReward(self):
        '''
        Calculates the reward for the warehouse based on the inventory cost.
        Returns:
        - The reward value.
  
        
        '''
        reward = 0
        reward -= self.Inventory * self.inventorycost[1]
        # print('www reward',reward,self.Inventory)
        return reward

    def _getObs(self):
        '''
        Internal method: Retrieves the observation state of the warehouse.
        Returns:
        - The inventory and a copy of the service queue.
  
        
        '''
        return self.Inventory, self.ServiceQueue.getQueue().copy()

    def nextAction(self, order):
        '''
        Determines the next action for the warehouse based on the order.
        Parameters:
        - order: The order value.
        Returns:
        - The observation state of the warehouse.
    
        '''

        self._updatServiceQueue()

        self.updateOrder(order)
        return self._getObs()


class Factory(BaseAgent):
    def __init__(self, warehouse: Warehouse, Inventory: int = 0, ServiceTime: int = 2, Demand: int = 0,
                 Freal_order: int = 0):
        '''
  
        Initializes an instance of the Factory class.
        Parameters:
        - warehouse: An instance of the Warehouse class representing the factory's warehouse.
        - Inventory: The initial inventory level of the factory (default: 0).
        - ServiceTime: The length of the service queue (default: 2).
        - Demand: The demand value (default: 0).
        - Freal_order: The real order value (default: 0).
        '''
        super(Factory, self).__init__()
        self.ServiceQueue = Queue(ServiceTime)
        # Initialize the service queue with zeros
        for i in range(ServiceTime):
            self.ServiceQueue.enqueue(0)
        self.Inventory = Inventory
        self.order = 0
        self.warehouse = warehouse
        self.FOrderlist = []
        self.Freal_order = Freal_order

        self.Demand = Demand

    def _updatServiceQueue(self):
        '''
        Internal method: Updates the service queue by adding a new demand value from the warehouse.
        
        '''
        self.Demand = self.warehouse.getOrderRequest()

        self.ServiceQueue.enqueue(self.Demand)

    def updateOrder(self, order):
        '''
        Updates the order value for the factory.
        Parameters:
        - order: The new order value.
        '''
        self.order = order

    def getOrderRequest(self) -> int:
        '''
        Retrieves the order request for the factory.
        Returns:
         - The order value if an action trigger is True, otherwise returns 0.
        
        '''
        if BaseAgent.actionTrigger:
            return self.order
        else:

            return 0

    def _updateInventory(self, preOrder):
        '''
        Internal method: Updates the factory's inventory based on the previous order.
        Parameters:
        - preOrder: The previous order value.
        '''

        self.Demand = self.ServiceQueue.dequeue()
        self.FOrderlist = np.append(self.FOrderlist, [self.Demand])

        self.Inventory += preOrder

        if self.Demand <= self.Inventory:
            self.Inventory -= self.Demand
            self.Wreal_order = self.Demand
            self.warehouse.updateInventory(self.Demand)
        else:
            self.Demand = self.Inventory
            self.Wreal_order = self.Inventory
            self.warehouse.updateInventory(self.Demand)
            self.Inventory = 0
        # print('self.Wreal_order ',self.Wreal_order )
        # print('**always true',self.Wreal_order == self.warehouse.preOrder)

        if self.Inventory > 30:
            self.Inventory = 30

        if self.Inventory < 0:
            self.Inventory = 0

    def getReward(self):
        '''
        Calculates the reward for the factory based on the inventory cost.
        Returns:
        - The reward value.
  
        '''
        reward = 0
        reward -= self.Inventory * self.inventorycost[2]
        # print('ff' ,reward, self.Inventory)

        return reward

    def _getObs(self):
        '''
        Internal method: Retrieves the observation state of the factory.
        Returns:
        - The inventory and a copy of the service queue.
        
        '''
        return self.Inventory, self.ServiceQueue.getQueue().copy()

    def nextAction(self, order):
        '''
        Determines the next action for the factory based on the order.
        Parameters:
        - order: The order value.
        Returns:
        - The observation state of the factory.
      
        
        '''
        self._updatServiceQueue()
        self.updateOrder(order)
        obs = self._getObs()

        return obs


class Supplier(BaseAgent):
    def __init__(self, factory: Factory, Inventory: int = 0, ServiceTime: int = 0, backout: int = 0,
                 urgentOrder: int = 0, Demand: int = 0):
        '''
        Initializes an instance of the Supplier class.
        Parameters:
        - factory: An instance of the Factory class .
        - Inventory: The initial inventory level of the supplier (default: 0).
        - ServiceTime: The length of the service queue (default: 0).
        - Demand: The demand value (default: 0).
        """
        '''
        super(Supplier, self).__init__()
        self.ServiceQueue = Queue(ServiceTime)
        # Initialize the service queue with zeros
        for i in range(ServiceTime):
            self.ServiceQueue.enqueue(0)

        self.order = 0
        self.factory = factory

        self.Demand = Demand

    def _updatServiceQueue(self):
        '''
        Internal method: Updates the service queue by adding a new demand value from the factory.
        
        '''
        self.Demand = self.factory.getOrderRequest()
        self.ServiceQueue.enqueue(self.Demand)
        BaseAgent.actionTrigger = False

    def _updateInventory(self):
        '''
        Internal method: Updates the supplier's GOODS based on the demand and updates the factory's inventory.
        '''

        self.factory._updateInventory(self.Demand)
        self.Demand = self.ServiceQueue.dequeue()

    def _getObs(self):
        '''
        Internal method: Retrieves the observation state of the supplier.
        Returns:
        - A copy of the service queue.
 
        '''
        return self.ServiceQueue.getQueue().copy()

    def nextAction(self):
        '''
        Determines the next action for the supplier.
        Returns:
        - The observation state of the supplier.
        '''

        self._updatServiceQueue()

        obs = self._getObs()
        self._updateInventory()

        return obs


class SupplyEnv(Env):
    def __init__(self, mean, std):
        '''
        Initializes an instance of the SupplyEnv class.
        '''
        low = np.array([10, 8, 8, 0], dtype=np.float32)
        high = np.array([10, 13, 13, 6], dtype=np.float32)
        self.action_space = spaces.Box(low=low, high=high, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=31, shape=(13,), dtype=np.float32)
        self.mean = mean
        self.std = std
        self.customer = Customer(self.mean, self.std )
        self.iters = 0
        self.retailer = Retailer(self.customer, Inventory=10, ServiceTime=0)
        self.warehouse = Warehouse(self.retailer, Inventory=0, ServiceTime=3)
        self.factory = Factory(self.warehouse, Inventory=0, ServiceTime=3)
        self.supplier = Supplier(self.factory, Inventory=0, ServiceTime=0)
        self.previous_action = [0, 0, 0, 5]
        self.reorder_point = 5
        self.retailer_order = 10
        self.num_envs = 1


        self.reset()

    def reset(self,seed=random_seed):
        super().reset(seed=seed)
        '''
        Resets the environment to its initial state.
        Returns:
        - The initial state.
        
        '''
        # print('@@@')
        global lives
        lives = 0
        self.iters = 0
        self.customer = Customer(self.mean, self.std)
        self.retailer = Retailer(self.customer, Inventory=10, ServiceTime=0)
        self.warehouse = Warehouse(self.retailer, Inventory=0, ServiceTime=3)
        self.factory = Factory(self.warehouse, Inventory=0, ServiceTime=3)
        self.supplier = Supplier(self.factory, Inventory=0, ServiceTime=0)

        RObs = self.retailer.nextAction(self.reorder_point)
        RState = np.array([RObs[0], *RObs[1]])

        WObs = self.warehouse.nextAction(0)
        WState = np.array([WObs[0], *WObs[1]])

        FObs = self.factory.nextAction(0)
        FState = np.array([FObs[0], *FObs[1]])

        SObs = self.supplier.nextAction()
        SState = np.array([SObs[0]])
        state = np.array([*RState,*WState,*FState,*SState],dtype=np.float32)
        return (state,{})



    def TotalReward(self):
        '''
        Calculates the total reward based on the rewards from the retailer, warehouse, and factory.
        Returns:
        - The total reward value.
        '''
        reward = 0
        Rreward = self.retailer.getReward()

        Wreward = self.warehouse.getReward()
        Freward = self.factory.getReward()

        reward = Rreward + Wreward + Freward
        # print('reward:',reward)

        return reward

    def step(self, action):
        '''
        Performs a step in the environment given an action.
        Parameters:
        - action: The action to be taken.
        Returns:
        - The next state, reward, done flag, and additional info.
  
        '''
        # print('action--',action)
        action = np.floor(action)
        # if (self.retailer.Inventory <= self.reorder_point) and (self.warehouse.ServiceQueue.getQueue().copy() == [0,0,0]) and (self.warehouse.WOrderlist[-1]==0):

        self.retailer.sandactionTrigger(self.warehouse, self.reorder_point)
        if BaseAgent.actionTrigger == True:
            self.reorder_point = action[3]
            action[0] = self.retailer_order
            self.retailer.stockout = 0
            warehouse_order = action[1]
            factory_order = action[2]



        else:
            # BaseAgent.actionTrigger = False
            action[0] = 0

            self.reorder_point = self.previous_action.copy()[3]
            action[3] = self.reorder_point
            action[1] = 0
            action[2] = 0

            warehouse_order = action[1]
            factory_order = action[2]

        info = {'actionTrigger': BaseAgent.actionTrigger}
        # print('--------------------------Step',self.iters,'--------------------------')

        reward = self.TotalReward()
        print('action',action)
        print('reward',reward)


        RObs = self.retailer.nextAction(self.reorder_point)

        RState = np.array([RObs[0], *RObs[1]])

        WObs = self.warehouse.nextAction(warehouse_order)

        WState = np.array([WObs[0], *WObs[1]])

        FObs = self.factory.nextAction(factory_order)

        FState = np.array([FObs[0], *FObs[1]])

        SObs = self.supplier.nextAction()
        SState = np.array([SObs[0]])

        state = np.array([*RState, *WState, *FState, *SState],dtype=np.float32)
        print(state)


        global lives

        terminated = lives>3
        truncated = self.iters>30

        # done = (self.iters > 30) or (lives > 3)

        self.previous_action = action
        self.iters += 1


        return state, reward, truncated, terminated, info


class LeanSupplyEnv0(SupplyEnv):
    def __init__(self, mean=2, std=0):
        super().__init__(mean, std)
        # self.observation_space = spaces.Box(low=0, high=30, shape=(13,), dtype=np.float32)

class LeanSupplyEnv01(SupplyEnv):
    def __init__(self, mean=2, std=0.1):
        super().__init__(mean, std)
        # self.action_space = CustomSpace()
        # self.action_space = spaces.Discrete(low=low,high = high,shape=(4,),dtype=np.int32)
        # self.observation_space = spaces.Box(low=0, high=31, shape=(13,), dtype=np.float32)
# class AgileSupplyEnv0(SupplyEnv):
#     def __init__(self, mean=2, std=0.1):
#         super().__init__(mean, std)
#         low = np.array([10, 0, 0, 1], dtype=np.float32)
#         high = np.array([10, 17, 17, 6], dtype=np.float32)
#         self.action_space = spaces.Box(low=low, high=high, shape=(4,), dtype=np.float32)


class AgileSupplyEnv0(SupplyEnv):
    def __init__(self, mean=2, std=1):
        super().__init__(mean, std)
        # low = np.array([10, 1, 1, 0], dtype=np.float32)
        # high = np.array([10, 17, 17, 6], dtype=np.float32)
        # self.action_space = spaces.Box(low=low, high=high, shape=(4,), dtype=np.float32)



def test():
    from stable_baselines3.common.env_checker import check_env
    env  = LeanSupplyEnv0()
    check_env(env)
    print(type(env.action_space) )
    #
    for i in range(3):
        action = env.action_space.sample()

        env.step(action)
        print(action)




if __name__=='__main__':


    test()