

# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 19:02:22 2023
@author: ww1a23
"""
import pandas as pd
import numpy as np
import os
import sys
current_dir = os.path.dirname(__file__)
sys.path.append(current_dir)
from utilss.utils import Queue
from gymnasium import Env,spaces
from common import BaseAgent
import random

#random_seed = 2023
#np.random.seed(random_seed)
#random.seed()
lives = 0
class Customer():
    def __init__(self,muDemand,stdDemand,batch_size,random_seed=None):
        self.index = 0
        self.batch_size = batch_size
        self.muDemand = muDemand
        self.stdDemand = stdDemand
        self.random_seed = random_seed
        self.generator = self.generate_data()
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
            
    def generate_data(self):
        while True:
            value = np.maximum(0,np.floor(np.random.normal(self.muDemand,self.stdDemand)))
            values = np.repeat(value,self.batch_size) 
            for value in values:
                yield value
    
    def get_batch(self,batch_size):
        batch = []
        for _ in range(batch_size):
            batch.append(next(self.generator))
        return batch
    
    def _getdemand(self):
        return next(self.generator)
    
    def getOrder(self):
      
        return self._getdemand()

# class Customer():
#     def __init__(self,muDemand=2,stdDemand=1):
#         self.muDemand = muDemand
#         self.stdDemand = stdDemand
#         self.index = 0
#     def _getDemand(self):
#         demand = max(0,np.random.normal(self.muDemand,self.stdDemand))
#         demand = np.floor(demand)
#         return demand
    
#     def getOrder(self):
#         return self._getDemand()


class Retailer(BaseAgent):
    def __init__(self,customer:Customer,Inventory:int=10,
                 ServiceTime:int=0,stockout:int=0,
                 Demand:int=0,stockoutcost:int=10000):
        super(Retailer,self).__init__()
        self.ServiceQueue = Queue(ServiceTime)
        for i in range(ServiceTime):
            self.ServiceQueue.enqueue(0)
        self.Inventory = Inventory
        self.order= 10
        self.customer = customer
        self.reorderPoint =5
        self.stockout = stockout
        self.Demand = Demand
        self.stockoutcost = stockoutcost
        self.preOrder = 0
        self.Sum_stockout=0
    
    def _updateServiceQueue(self):
        self.Demand = self.customer.getOrder()
        # print('Demand',self.Demand)
        self.ServiceQueue.enqueue(self.Demand)
    def updatereorderPoint(self,reorderPoint):
        self.reorderPoint = reorderPoint
    
    def _isOrder(self,other:BaseAgent)->bool:
        # print((self.Inventory<=self.reorderPoint) and (other.ServiceQueue.getQueue().copy()==[0,0,0]) and (other.WOrderlist[-1]==0) )
        return (self.Inventory<=self.reorderPoint) and (other.ServiceQueue.getQueue().copy()==[0,0,0]) and (other.WOrderlist[-1]==0)  
 
    def sandactionTrigger(self,other:BaseAgent,reorderPoint):
        self.reorderPoint = reorderPoint
        if self._isOrder(other):
            BaseAgent.actionTrigger= True
        else:
            BaseAgent.actionTrigger= False
    def getOrderRequest(self)->int:
        if BaseAgent.actionTrigger:
            return self.order
        else:
            return 0
    def updateInventory(self,preOrder):
        self.Demand = self.ServiceQueue.dequeue()
        self.preOrder = preOrder
        self.Inventory += self.preOrder
        if self.Demand <= self.Inventory:
            self.Inventory -= self.Demand
            # self.CReal_order = self.Demand
        else:
            self.stockout += self.Demand - self.Inventory
            global lives
            lives += 1
            # print('lives left',3-lives)
            # self.Demand = self.Inventory
            # self.CReal_order = self.Inventory
            self.Inventory = 0
        # if BaseAgent.actionTrigger:
        #     self.stockout = 0

        self.Inventory = max(0,min(self.Inventory,30))
        
    def getReward(self):
        reward = 0
        reward -= self.Inventory*self.inventorycost[0]
        reward -= self.stockout*self.stockoutcost
        return reward
    
    def _getObs(self):
        return self.Inventory,self.ServiceQueue.getQueue().copy()
    
    def nextAction(self,reorderPoint):
        self.updatereorderPoint(reorderPoint)
        self._updateServiceQueue()
        return self._getObs()
    
class Warehouse(BaseAgent):
    def __init__(self,retailer:Retailer, Inventory:int=0,
                 ServiceTime:int=3, Demand:int=0,RReal_order:int=0):
        super(Warehouse,self).__init__()
        self.ServiceQueue = Queue(ServiceTime)
        for i in range(ServiceTime):
            self.ServiceQueue.enqueue(0)
        self.Inventory = Inventory
        self.preOrder = 0
        self.order = 0
        self.retailer = retailer
        self.WOrderlist = []
        self.RReal_order = RReal_order
        self.Demand = Demand
    
    
    def _updateServiceQueue(self):
        self.Demand = self.retailer.getOrderRequest()
        self.ServiceQueue.enqueue(self.Demand)
    
    def updateOrder(self,order):
        self.order = order
    def getOrderRequest(self)->int:
        if BaseAgent.actionTrigger:
            return self.order
        else:
            return 0
    def updateInventory(self,preOrder):
        self.preOrder = preOrder
        self.Demand = self.ServiceQueue.dequeue()
        self.WOrderlist = np.append(self.WOrderlist,[self.Demand])

        self.Inventory += self.preOrder
        if self.Demand <= self.Inventory:
            self.Inventory -= self.Demand
            self.RReal_order = self.Demand
            self.retailer.updateInventory(self.Demand)
        else:
            self.Demand = self.Inventory
            self.RReal_order = self.Inventory
            self.retailer.updateInventory(self.Demand)
            self.Inventory = 0
            
        self.Inventory = max(0,min(self.Inventory,30))
        
    def getReward(self):
        reward = 0
        reward -=self.Inventory*self.inventorycost[1]
        return reward
    
    def _getObs(self):
        return self.Inventory,self.ServiceQueue.getQueue().copy()
    
    def nextAction(self,order):
        self._updateServiceQueue()
        self.updateOrder(order)
        return self._getObs()
    
class Factory(BaseAgent):
    def __init__(self,warehouse:Warehouse,Inventory:int=0,ServiceTime:int=2,
                 Demand:int=0,FReal_order:int=0):
        super(Factory,self).__init__()
        self.ServiceQueue = Queue(ServiceTime)
        for i in range(ServiceTime):
            self.ServiceQueue.enqueue(0)
        self.Inventory = Inventory
        self.order = 0
        self.warehouse = warehouse
        self.FOrderlist = []
        self.FReal_order= FReal_order
        self.Demand = Demand
    
    def _updateServiceQueue(self):
        self.Demand = self.warehouse.getOrderRequest()
        self.ServiceQueue.enqueue(self.Demand)
        
    def updateOrder(self,order):
        self.order = order
        
    def getOrderRequest(self)->int:
        if BaseAgent.actionTrigger:
            return self.order
        else:
            return 0
    def _updateInventory(self,preOrder):
        self.Demand = self.ServiceQueue.dequeue()
        self.FOrderlist = np.append(self.FOrderlist,[self.Demand])
        self.Inventory += preOrder
        if self.Demand <= self.Inventory:
            self.Inventory -= self.Demand
            self.WReal_order = self.Demand
            self.warehouse.updateInventory(self.Demand)
        else:
            self.Demand = self.Inventory
            self.WReal_order = self.Inventory
            self.warehouse.updateInventory(self.Demand)
            self.Inventory = 0
        self.Inventory = max(0,min(self.Inventory,30))
    
    def getReward(self):
        reward = 0
        reward -= self.Inventory*self.inventorycost[2]
        return reward
    
    def _getObs(self):
        return self.Inventory,self.ServiceQueue.getQueue().copy()
    
    def nextAction(self,order):
        self._updateServiceQueue()
        self.updateOrder(order)
        obs = self._getObs()
        return obs
    
class Supplier(BaseAgent):
    def __init__(self,factory:Factory,Inventory:int=0,ServiceTime:int=0,
                 backout:int=0,urgentOrder:int=0,Demand:int=0):
        super(Supplier,self).__init__()
        self.ServiceQueue = Queue(ServiceTime)
        for i in range(ServiceTime):
            self.ServiceQueue.enqueue(0)
        self.order = 0
        self.factory = factory
        self.Demand = Demand
    
    def _updateServiceQueue(self):
        self.Demand = self.factory.getOrderRequest()
        self.ServiceQueue.enqueue(self.Demand)
        BaseAgent.actionTrigger = False  # duan dian
        
    def _updateInventory(self):
        self.factory._updateInventory(self.Demand)
        self.Demand = self.ServiceQueue.dequeue()
        
    def _getObs(self):
        return self.ServiceQueue.getQueue().copy()
    
    def nextAction(self):
        self._updateServiceQueue()
        obs = self._getObs()
        self._updateInventory()
        return obs
    
    
            
            
        
        
class SupplyEnv(Env):
    def __init__(self,muDemand,stdDemand,batch_size):
        #low = np.array([8,8,8,1],dtype=np.float32)
        #high = np.array([10,17,17,6],dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([10, 30, 30, 5])
        #self.action_space = spaces.Box(low=low,high=high,shape=(4,),dtype=np.float32)
        self.observation_space = spaces.Box(low=0,high=31,shape=(13,),dtype=np.float32)
        self.muDemand = muDemand
        self.stdDemand = stdDemand
        self.batch_size = batch_size
        self.customer = Customer(self.muDemand,self.stdDemand,self.batch_size)
        self.iters =0
        self.retailer = Retailer(self.customer,Inventory=10,ServiceTime=0)
        self.warehouse = Warehouse(self.retailer,Inventory=0,ServiceTime=3)
        self.factory = Factory(self.warehouse,Inventory=0,ServiceTime=3)
        self.supplier = Supplier(self.factory,Inventory=0,ServiceTime=0)
        self.previous_action = [0,0,0,5]
        self.reorder_point = 5
        self.retailer_order = 10
        self.num_envs = 1  
        self.total_rew = 0
        self.reset()
    def reset(self,seed=2023):
        super().reset(seed=seed)
        global lives
        lives = 0
        self.iters = 0
        self.total_rew = 0
        self.customer = Customer(self.muDemand,self.stdDemand,self.batch_size)
        self.retailer = Retailer(self.customer, Inventory=10, ServiceTime=0)
        self.warehouse = Warehouse(self.retailer, Inventory=0, ServiceTime=3)
        self.factory = Factory(self.warehouse,Inventory=0,ServiceTime=3)
        self.supplier = Supplier(self.factory,Inventory=0,ServiceTime=0)
        RObs = self.retailer.nextAction(self.reorder_point)
        RState = np.array([RObs[0],*RObs[1]])
        WObs = self.warehouse.nextAction(0)
        WState = np.array([WObs[0],*WObs[1]])
        FObs = self.factory.nextAction(0)
        FState = np.array([FObs[0],*FObs[1]])
        SObs = self.supplier.nextAction()
        SState = np.array([SObs[0]])
        state = np.array([*RState,*WState,*FState,*SState],dtype=np.float32)
        return (state,{})
        # return state
    
    def TotalReward(self):
        reward = 0
        # global lives
        # lives = 0
        Rreward = self.retailer.getReward()
        Wreward = self.warehouse.getReward()
        Freward = self.factory.getReward()
        reward = Rreward + Wreward + Freward
        return reward
    
    def step(self,action):
        action = np.floor(action)
        self.retailer.sandactionTrigger(self.warehouse,self.reorder_point)
        if BaseAgent.actionTrigger == True:
            self.reorder_point = action[3]
            action[0] = self.retailer_order
            self.retailer.stockout = 0
            warehouse_order = action[1]
            factory_order = action[2]
        else:
            action[0] = 0 
            self.reorder_point = self.previous_action.copy()[3]
            action[3] = self.reorder_point
            action[1]=0     
            action[2]=0
            warehouse_order =action[1]
            factory_order = action[2]
            
        info={'actionTrigger':BaseAgent.actionTrigger}
        # print('-------------------------------', self.iters,'-------------------------------')
        reward = self.TotalReward()
        self.total_rew += reward
        # print(info)
        
        # print(action,reward) 
        RObs = self.retailer.nextAction(self.reorder_point)
        
        RState = np.array([RObs[0],*RObs[1]])
        WObs = self.warehouse.nextAction(warehouse_order) 
        WState = np.array([WObs[0],*WObs[1]])
        FObs = self.factory.nextAction(factory_order)  
        FState = np.array([FObs[0],*FObs[1]])
        SObs = self.supplier.nextAction()
        SState = np.array([SObs[0]])
        state = np.array([*RState,*WState,*FState,*SState],dtype=np.float32)
        # print(state)
        global lives
        terminated =lives>3
        truncated = self.iters>30
        # done = (self.iters>30) or (lifes>30)
        self.previous_action=action
        self.iters+=1
        #print(state)
        # print(action)
        # print(reward)
        
        
        return state,reward,terminated,truncated,info
    
    @property
    def get_episode_reward(self):
        return self.total_rew
    @property
    def get_episode_length(self):
        return self.iters
 

   
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
            
# def test():
#     from stable_baselines3 import PPO
#     from stable_baselines3.common.evaluation import evaluate_policy  
#     result={}
#     stds= [0,0.01,0.1,1]
    
#     std_rewards=[]
#     mean_rewards =[]
#     game_steps=list(range(10,310,10))
#     for std in stds:
#         env = SupplyEnv(mean=2,std=std)
#         ppo_agent = PPO('MlpPolicy',env=env,verbose=2)
#         print('---------@@@@@@--------',std)
#         obs=env.reset()
        
#         is_done = False
#         state = env.reset()
#         # old_reward = -10000
#         means = []
#         steptimes = 2
#         ppo_agent.learn(total_timesteps=1000)
#         ppo_agent.save('ppo_sc')
#         del ppo_agent
#         model=PPO.load('ppo_sc',env=env)
        # mean_reward,std_reward=evaluate_policy(model,env,n_eval_episodes=1000)
    #     mean_rewards.append(np.mean(mean_reward))
    #     std_rewards.append(np.std(std_reward))
    #     print('mean_reward',mean_rewards)
    #     print('std_reward',std_rewards)
    #     result['std={}'.format(std)]=mean_rewards
    # result['Different rewards each time based on the std']=game_steps
    # result = pd.DataFrame(result)
    # result.set_index('Different rewards each time based on the std',drop=True,inplace=True)
    # result.plot.line(figsize=(10,5))
    # plt.show()
        
#         for s in game_steps:
#             # action = env.action_space.sample()
#             action,_state = model.predict(obs)
            
#             old_state = state
#             new_state,current_reward,is_done,info = env.step(action)
#             print('current_reward',current_reward)
#             print('action',action)
            
#             # means.append(np.mean(current_reward))
#             # print('means',means)
            
#             state = env.reset() if is_done else new_state
            
#             print('state:',new_state)
#             # print('old_state:',old_state)
    
 
#             if is_done:
#                 break
            
#             print('--------------------------Step',steptimes,'--------------------------')
#             steptimes += 1

    
# if __name__=='__main__':
#     test()
            
# def test():
#     from stable_baselines3 import PPO
#     from stable_baselines3.common.evaluation import evaluate_policy  
#     env = SupplyEnv()
#     ppo_agent = PPO('MlpPolicy',env=env,verbose=2)
    
#     result={}
#     for i in range(1):
        
#         obs=env.reset()
#         is_done = False
#         # state = env.reset()
#         old_reward = -10
#         # steptimes = 2
#         ppo_agent.learn(total_timesteps=10)
#         ppo_agent.save('ppo_sc')
#         del ppo_agent
#         model=PPO.load('ppo_sc',env=env)
#         mean_reward,std_reward=evaluate_policy(model,env,n_eval_episodes=10)
        
#         # while True
#         means=[]
#         for i in range(30):
#             # action = env.action_space.sample()
#             # old_state = state
#             action,_state = model.predict(obs)
#             new_state,current_reward,is_done,info = env.step(action)
#             means.append(np.mean(current_reward))
#             print('new_state',new_state)
#             print('current_reward',current_reward)
#             print('is_done',is_done)
#         result['epsilon={}'.format(i)]=means
#         result['step']=
            
#             # state = env.reset() if is_done else new_state
            
#             # print('state:',new_state)
#             # print('old_state:',old_state)
    
 
#             # if is_done:
#             #     break
            
#             # print('--------------------------Step',steptimes,'--------------------------')
#             # steptimes += 1
    
# if __name__=='__main__':
#     test()


    
class LeanSupplyEnv1(SupplyEnv):
    def __init__(self,muDemand=2,stdDemand=0.1,batch_size=10):
        super().__init__(muDemand,stdDemand,batch_size)
        # self.observation_space = spaces.Box(low=0,high=31,shape=(13,),dtype=np.int32)
    
class LeanSupplyEnv2(SupplyEnv):
    def __init__(self, mean=2, std=0.1,batch_size=7):
        super().__init__(mean, std,batch_size)
#         self.observation_space = spaces.Box(low=0,high=31,shape=(13,),dtype=np.int32)
# class AgileSupplyEnv0(SupplyEnv):
#     def __init__(self, mean=2, std= 0.1):
#         super().__init__(mean, std)
#         low = np.array([10, 0, 0, 0], dtype=np.int32)
#         high = np.array([10, 17, 17, 6], dtype=np.int32)      
#         self.action_space = spaces.Box(low=low,high = high,shape=(4,),dtype=np.int32)
        
class AgileSupplyEnv3(SupplyEnv):
    def __init__(self,  muDemand=2,stdDemand=0.1,batch_size=3):
        super().__init__( muDemand,stdDemand,batch_size)
        # low = np.array([10, 1, 1, 0], dtype=np.int32)
        # high = np.array([10, 17, 17, 6], dtype=np.int32)      
        # self.action_space = spaces.Box(low=low,high = high,shape=(4,),dtype=np.int32)


def test(): 
    env = SupplyEnv(muDemand=2,stdDemand=0.1,batch_size=3) 
    pisode_reward = env.get_episode_reward
    print('ppppp',pisode_reward)
    
    for i in range(1): 
        obs=env.reset() 
        is_done = False 
  
        for i in range(80): 
            action = env.action_space.sample() 

            new_state,current_reward,terminated,truncated,info = env.step(action)
            if (is_done):
                break
            print('new_state',new_state) 
            print('current_reward',current_reward) 
            print('is_done',is_done) 

if __name__=='__main__': 
    test()


        
             
