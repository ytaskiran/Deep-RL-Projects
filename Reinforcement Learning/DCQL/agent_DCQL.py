import random
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Conv2D
from collections import deque

NUM_ACTIONS = 3
IMGHEIGHT = 40
IMGWIDTH = 40
IMGHISTORY = 4

OBSERVE_PERIOD = 2000
GAMMA = 0.975
BATCH_SIZE = 64
EXPREPLAY_CAPACITY = 2000

class Agent:
    def __init__(self):
        self.model = self.buildModel()
        self.expReplay = deque()
        self.steps = 0
        self.epsilon = 1
    
    def buildModel(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=4, strides=(2,2), input_shape=(IMGHEIGHT,IMGWIDTH,IMGHISTORY), padding = "same"))
        model.add(Activation("relu"))
        model.add(Conv2D(64, kernel_size=4, strides=(2,2), padding="same"))
        model.add(Activation("relu"))
        model.add(Conv2D(64, kernel_size=4, strides=(2,2), padding="same"))
        model.add(Activation("relu"))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation("relu"))
        model.add(Dense(units=NUM_ACTIONS, activation="linear"))
        
        model.compile(loss="mse", optimizer="adam")
        
        return model
        
    def findBestAct(self, state):
        if random.random() < self.epsilon or self.steps < OBSERVE_PERIOD:
            return random.randrange(0,NUM_ACTIONS)
        else:
            qvalues = self.model.predict(state)
            return np.argmax(qvalues)
    
    def captureSample(self, sample):
        self.expReplay.append(sample)
        if len(self.expReplay) > EXPREPLAY_CAPACITY:
            self.expReplay.popleft()
            
        self.steps += 1
        
        if self.steps > OBSERVE_PERIOD and self.steps <= 6000:
            self.epsilon = 0.75
        elif self.steps > 6000 and self.steps <= 12000:
            self.epsilon = 0.5
        elif self.steps > 12000 and self.steps <= 25000:
            self.epsilon = 0.25
        elif self.steps > 25000 and self.steps <= 40000:
            self.epsilon = 0.15
        elif self.steps > 40000 and self.steps <= 60000:
            self.epsilon = 0.1
        elif self.steps > 60000:
            self.epsilon = 0.05            
            
    def process(self):
        if self.steps > OBSERVE_PERIOD:
            minibatch = random.sample(self.expReplay, BATCH_SIZE)
            inputs = np.zeros((BATCH_SIZE, IMGHEIGHT, IMGWIDTH, IMGHISTORY))
            targets = np.zeros((inputs.shape[0], NUM_ACTIONS))
            Q = 0
            
            for i in range(len(minibatch)):
                state_t = minibatch[i,0]
                action_t = minibatch[i,1]
                reward_t = minibatch[i,2]    
                state_t1 = minibatch[i,3]
                
                inputs[i:i+1] = state_t
                targets[i] = self.model.predict(state_t)
                Q = self.model.predict(state_t1)
                
                if state_t1 is None:
                    targets[i,action_t] = reward_t
                else:
                    targets[i,action_t] = reward_t + GAMMA*np.max(Q)
              
            self.model.fit(inputs, targets, batch_size=BATCH_SIZE, epochs=1, verbose=0)
            


















