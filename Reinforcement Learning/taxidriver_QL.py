# %%
import gym
import numpy as np
import random
import matplotlib.pyplot as plt

env = gym.make("Taxi-v3").env

#Q table
qTable = np.zeros([env.observation_space.n, env.action_space.n])

#Hyperparameters
alpha = 0.1
gamma = 0.9
epsilon = 0.1

#Lists for visualization 
rewardList = []
dropoffList = []

episodeNum = 2000

for i in range(1,episodeNum):
    
    #initialize env
    state = env.reset()
    
    rewardCounter = 0
    doffCounter = 0
    
    while True:
        
        #exploit vs explore to find action
        if random.uniform(0,1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(qTable[state])
        
        #action process and take reward/observation
        nextState, reward, done, _ = env.step(action)
        
        #Q learning funct
        oldValue = qTable[state, action]
        nextMax = np.max(qTable[nextState])
        newValue = (1-alpha)*oldValue + alpha*(reward+ gamma*nextMax)
        
        #update q table 
        qTable[state, action] = newValue
        
        #update state
        state = nextState
        
        #find wrong dropoffs
        if reward == -10:
            doffCounter += 1
            
        rewardCounter += reward
        
        if done:
            break
    
    if i%10 == 0:
        dropoffList.append(doffCounter)
        rewardList.append(rewardCounter)
        print("Episode: {}, Reward: {}, Wrong Dropoffs: {}".format(i, rewardCounter, doffCounter))

# %%
  
fig, axes = plt.subplots(1,2)
axes[0].plot(rewardList)
axes[0].set_xlabel("episode")
axes[0].set_ylabel("reward")

axes[1].plot(dropoffList)
axes[1].set_xlabel("episode")
axes[1].set_ylabel("number of wrong dropoffs")

axes[0].grid(True)
axes[1].grid(True)

plt.show()

        


