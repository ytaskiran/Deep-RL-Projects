# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 16:19:19 2019

@author: ytask
"""
# %%
import gym 
import random 
import numpy as np

#import env
env = gym.make("FrozenLake-v0").env

#from gym.envs.registration import register
#register(
#        id = "FrozenLakeNotSlippery-v0",
#        entry_point="gym.envs.toy_test:FrozenLakeEnv",
#        kwargs={"map_name" : "4x4", "is_slippery" : False},
#        max_episode_steps=100,
#        reward_threshold=0.78)

#create QTable
qTable = np.zeros([env.observation_space.n, env.action_space.n])

#hyperparameters
alpha = 0.12
gamma = 0.99

exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.001

rewList = []
episodeNum = 8000

for episode in range(1, episodeNum+1):
    
    state = env.reset()    
    rewCounter = 0
   
    while True:
        
        if random.uniform(0,1) < exploration_rate:
            action = env.action_space.sample()
        else:
            action = np.argmax(qTable[state])
            
        #action process
        newState, reward, done, _ = env.step(action)
        
        #qlearning func
        qOld = qTable[state, action]
        maxNext = np.max(qTable[newState])
        qNew = (1-alpha)*qOld + alpha*(reward+gamma*maxNext)
        
        #update qtable
        qTable[state, action] = qNew
        state = newState
        rewCounter += reward
                       
        if done:
            break
    exploration_rate = min_exploration_rate + \
        (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*episode)    
    
    rewList.append(rewCounter)   
    if episode%100 == 0:
        print("Episode: {}, Reward: {}".format(episode,rewCounter))
        
rewards_per_thosand_episodes = np.split(np.array(rewList),episodeNum/1000)
count = 1000

print("********Average reward per thousand episodes********\n")
for r in rewards_per_thosand_episodes:
    print(count, ": ", str(sum(r/1000)))
    count += 1000
        
        
# %%
import time
from IPython.display import clear_output

for episode in range(5):
    # initialize new episode params
    state = env.reset()
    done = False
    print("*****EPISODE ", episode+1, "*****\n\n\n\n")
    time.sleep(1)
    for step in range(100):        
        # Show current state of environment on screen
        #clear_output(wait = True)
        env.render()
        time.sleep(0.5)
        
        # Choose action with highest Q-value for current state 
        action = np.argmax(qTable[state])
        # Take new action
        newState, rew, done, _ = env.step(action)
        
        state = newState
        
        if done:
            #clear_output(wait=True)
            env.render()
            if reward == 1:
                # Agent reached the goal and won episode
                print("****You reached the goal!****")
                time.sleep(3)
            else:
                # Agent stepped in a hole and lost episode 
                print("****You fell through a hole!****")
                time.sleep(3)
                clear_output(wait=True)
            break
        # Set new state
        
env.close()        
        
        
        
        
        
        
        
        
        