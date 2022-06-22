# -*- coding: utf-8 -*-
"""
Created on Sat May  7 17:47:29 2022

@author: Vinay Mandre
"""

import numpy as np
import random 
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

#creating architecture of Neural network
#for our problem, input_size i.e. input layers are 5 and output layers i.e. nb_action is 3
class Network(nn.Module):
    
    
   #following two classes are Deep Learning modules 
    def __init__(self, input_size, nb_action):
       super(Network, self).__init__()
       self.input_size = input_size
       self.nb_action=nb_action
       self.fc1 = nn.Linear(input_size, 30)   #all the input layers will be connected to all the hidden layers
       self.fc2 = nn.Linear(30,nb_action)      #all the hidden layers will be connected to the output layer
        
   #function to perform forward propagation of the above created neural network,  it activates the neural network and it will return q values each possible action
    def forword(self, state):
       x = F.relu(self.fc1(state))               #activation function that activates neuron ( rectifier function)
       q_values = self.fc2(x)
       return q_values
   
   
#experience replays
#storing data for future references 
class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []    #creating a memory list to store all the data
        
        
    #adding records to the memory list    
    def push(self, event):              
        self.memory.append(event)
        
        if len(self.memory) > self.capacity:        #if memory list is full delete first entry
            del self.memory[0]
        
    def sample(self, batch_size):
        
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x, 0)),samples)      #when agent comes back to same state it fetches random previosly stored data
   
    
#implementing Deep Q learning

class Dqn():
    
    def __init__(self, input_size, nb_action, gamma):
        
        self.gamma = gamma
        self.reward_window = []
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(100000)
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)  #lr = learning rate
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0
        
    def select_action(self, state):      #softmax function
        probs = F.softmax(self.model(Variable(state, volatile = True))*0)  
        #T-Temperature Parameter = 7
        #Temperature parameter - this will increase the value of output of softmax function(i.e. highest probability value which means which action should be taken) 
        #this is done because agent will know exactly which action is should perform
       
        action = probs.multinomial()
        return action.data[0,0]
   
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = self.model(batch_state).gather(1,batch_action).unsqueeze(1).squeeze(1)
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma*next_outputs + batch_reward #calculating loss function
        td_loss = F.smooth_l1_loss(outputs,target)  
        self.optimizer.zero_grad()
        td_loss.backward(retails_variables = True)  #backpropagation 
        self.optimizer.step()                       #updates the weights
        
    def update(self,reward,new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push((self.last_state, new_state, torch.LongTensor( [int(self.last_action)]),torch.Tensor([self.last_reward])))
        action = self.select_action(new_state)
        
        if len(self.memory.memory) > 100:
            
            batch_state, batch_next_state, batch_reward, batch_action = self.memory.sample(100)  #learning from 100 entries
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        
        self.last_action = action
        self.last_state= new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        
        if len(self.reward_window) > 1000:
            del self.reward_window[0]                           
        
        return action
    
    
    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1.) 
    
    
    def save(self):
        
        torch.save({'state_dict' : self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict,
                    }, 'last_brain.pth')
    
    def load(self):
        
        if os.path.isfile('last_brain.pth'):
            print('Loading the model......')
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print('Done !')
        
        else:
            print('No such checkpoint found..... ')
        