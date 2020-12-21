import numpy as np
import random
import gym
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import model_from_json


class Agent:
    
    def __init__(self, env, epsilon=1):
        self.env = env
        self.stateSize = env.observation_space.shape[0]
        self.actionSize = env.action_space.n
        self.learningRate = 0.0003
        self.gamma = 0.99
        self.epsilon = epsilon
        self.minEpsilon = 0.005
        self.maxEpsilon = 1
        self.epsilonDecay = 0.002
        self.model = self.buildModel()
        self.target_model = self.buildModel()
        self.memory = deque(maxlen=4000)
        
    
    def selectAct(self, state, explored, exploited):
        if random.uniform(0,1) < self.epsilon:
            
            return self.env.action_space.sample(), explored+1, exploited
        else:
            actionValues = self.model.predict(state)
            return np.argmax(actionValues), explored, exploited+1
        
    
    def memorize(self, state, action, reward, nextState, done):
        self.memory.append((state, action, reward, nextState, done))
    
    def adaptiveEpsilon(self, e):
        if self.epsilon > self.minEpsilon:
            self.epsilon = self.minEpsilon + \
        (self.maxEpsilon - self.minEpsilon) * np.exp(-self.epsilonDecay*e)   
    
    def buildModel(self):
        model = Sequential()
        model.add(Dense(units=64, input_dim=self.stateSize, activation="relu"))
        model.add(Dense(units=64, activation="relu"))
        model.add(Dense(units=self.actionSize, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(learning_rate=self.learningRate))
        return model
    
#    def train(self, bs):
#        if len(self.memory) > bs:
#            minibatch = random.sample(self.memory, bs)
#            minibatch = np.array(minibatch)
#            
#            for state, nextState, reward, done in minibatch:
#                if done:
#                    target = reward
#                else:
#                    target = reward + np.amax(self.model.predict(nextState))
#                finalTargets = self.model.predict(state)
#                finalTargets[0,action] = target
#                self.model.fit(state, finalTargets, verbose=0)
        
    #vector way -- faster
    def train(self,batch_size):
        "vectorized replay method"
        if len(agent.memory) < batch_size:
            return
        # Vectorized method for experience replay
        minibatch = random.sample(self.memory, batch_size)
        minibatch = np.array(minibatch)
        not_done_indices = np.where(minibatch[:, 4] == False)
        y = np.copy(minibatch[:, 2])

        # If minibatch contains any non-terminal states, use separate update rule for those states
        if len(not_done_indices[0]) > 0:
            predict_sprime = self.model.predict(np.vstack(minibatch[:, 3]))
            predict_sprime_target = self.target_model.predict(np.vstack(minibatch[:, 3]))
            
            # Non-terminal update rule
            y[not_done_indices] += np.multiply(self.gamma, predict_sprime_target[not_done_indices, np.argmax(predict_sprime[not_done_indices, :][0], axis=1)][0])

        actions = np.array(minibatch[:, 1], dtype=int)
        y_target = self.model.predict(np.vstack(minibatch[:, 0]))
        y_target[range(batch_size), actions] = y
        self.model.fit(np.vstack(minibatch[:, 0]), y_target, epochs=1, verbose=0)   
        
    def targetModelUpdate(self):
        self.target_model.set_weights(self.model.get_weights())
        
      
    def saveModel(self):
        model_json = self.model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights("model.h5")
        print("Saved model to disk")
        
    def loadModel(self):
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("model.h5")
        self.model = loaded_model
        print("Loaded model from disk")
 
# %%       
if __name__ == "__main__":
    #importing env and agent
    env = gym.make("LunarLander-v2")
    agent = Agent(env)
    
    #episode and batchsize
    episode = 5000
    batchSize = 32
    
    #episodes
    for e in range(1,episode+1):
        state = env.reset()
        state = state.reshape([1,8])
        
        rew = 0
        explored = 0
        exploited = 0
        
        while True:
            env.render()
            action, explored, exploited = agent.selectAct(state, explored, exploited)
            nextState, reward, done, _ = env.step(action)
            nextState = nextState.reshape([1,8])
            agent.memorize(state, action, reward, nextState, done)
            state = nextState
            agent.train(batchSize)
            
            rew += reward
            
            if done:
                print("Episode {}, reward {}, explored {}, exploited {}".format(e,rew,explored,exploited))
                agent.targetModelUpdate()
                break
        
        agent.adaptiveEpsilon(e)        

# %% test
import time

env = gym.make("LunarLander-v2")
#trainedModel1 = agent 
trainedModel = Agent(env, 0)
trainedModel.loadModel()
 
for i in range(10): 
    state = env.reset()
    state = state.reshape([1,8])

    while True:
        env.render()
        action,_,_ = trainedModel.selectAct(state,0,0)
        nextState, reward, done, _ = env.step(action)
        nextState = nextState.reshape([1,8])
        state = nextState
        time.sleep(0.01)
        if done:
            break
        
    
    
    
    
            















