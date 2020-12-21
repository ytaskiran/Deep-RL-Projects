import numpy as np
import gym
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import random

class Agent(object):
    def __init__(self, env):
        #hyperparameters and parameters
        self.stateSize = env.observation_space.shape[0]
        self.actionSize = env.action_space.n
        self.gamma = 0.95
        self.learningRate = 0.0007
        self.epsilon = 1
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.memory = deque(maxlen = 1000)
        self.model = self.buildModel()
        
    def buildModel(self):
        #neural network 
        model = Sequential()
        model.add(Dense(48, input_dim=self.stateSize, activation="tanh"))
        model.add(Dense(units=self.actionSize, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(lr=self.learningRate))
        return model
    
    def remember(self, state, action, reward, nextState, done):
        #storage
        self.memory.append((state, action, reward, nextState, done))
    
    def act(self, state):
        #acting, explore or exploit
        if random.uniform(0,1) < self.epsilon:
            return env.action_space.sample()
        else:
            act_values = self.model.predict(state)
            return np.argmax(act_values)
    
    def replay(self, batchSize):
        #trainin
        if len(self.memory) < batchSize:
            return
        else:
            minibatch = random.sample(self.memory, batchSize)
            for state, action, reward, nextState, done in minibatch:
                if done:
                    target = reward
                else:
                    target = reward + self.gamma*np.amax(self.model.predict(nextState))
                trainTarget = self.model.predict(state)
                trainTarget[0,action] = target
                self.model.fit(state, trainTarget, verbose=0)
                    
                    
    def adaptiveEpsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon*self.epsilon_decay
        else:
            print("Epsilon is min!!")
        
    
    
if __name__ == "__main__":
    
    #initialize env and agent
    env = gym.make("CartPole-v1")
    agent = Agent(env)
    batchSize = 32
    episode = 200
    for e in range(episode):
        #initialize env
        state = env.reset()
        state = np.reshape(state, [1,4])
        time = 0
        
        while True:
            env.render()
            #selecting action
            action = agent.act(state)
            #step
            nextState, reward, done, _ = env.step(action)
            nextState = nextState.reshape([1,4])
            #remember
            agent.remember(state, action, reward, nextState, done)
            #update state
            state = nextState
            #replay
            agent.replay(batchSize)
            #adjust epsilon
            
            time += 1
            if done == True:
                agent.adaptiveEpsilon()
                print("Episode {}, time {}".format(e,time))
                break
            
            
 # %%
import time
trainedModel = agent
state = env.reset()
state = state.reshape([1,4])
time1 = 0    
while True:
    env.render()
    action = trainedModel.act(state)
    nextState, reward, done, _ = env.step(action)
    nextState = nextState.reshape([1,4])
    state = nextState
    time1 += 1
    print(time1)
    time.sleep(0.05)
    if done:
        break

            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            

        
    
    
    
    
    
    
    
    
    
    
    
    