import pandas as pd
import gym
import numpy as np

class Queue():
    def __init__(self,size):
        self.size = size+1
        self.front = -1
        self.rear = -1
        self.queue = []

    def enqueue(self,ele):
        if self.isfull():
            raise Exception('queue is full')
        
        else:
            self.queue.append(ele)
            self.rear = self.rear+1
    def dequeue(self):
        if self.isempty():
            raise Exception()
        else:
            self.front = self.front+1
            return self.queue.pop(0)
    
    def isfull(self):
        return self.rear - self.front == self.size
    def isempty(self):
        return self.front == self.rear
    def showQueue(self):
        print(self.queue)
    def getQueue(self):
        return self.queue
    


class CustomSpace(gym.Space):
    def __init__(self):
        self.n = 4
        self.dtype = np.int64
    def sample(self):
        # Randomly sample the values for the space
        value1 = 10
        value2 = int(np.random.normal(10, 1))
        value3 = int(np.random.normal(10, 1))
        value4 = np.random.randint(0, 7)  # The range is [0, 6]
        return np.array([value1, value2, value3, value4], dtype=self.dtype)
    def contains(self, x):
        # Check if the input x is within the space
        return (
            isinstance(x, (np.ndarray, np.generic)) and
            x.shape == (4,) and
            x.dtype.kind in 'iu' and
            np.all(x[0] == 10) and  # Check if the first value is 10
            0 <= x[3] <= 6  # Check if the fourth value is within [0, 6]
        )
    def __repr__(self):
        return "CustomSpace()"
# Create the custom space
# custom_space = CustomSpace()
# # Sample an action from the custom space
# action = custom_space.sample()
# print("Selected action:", action)

    


def getdata():
    data = pd.read_csv('./data.csv')
    return data