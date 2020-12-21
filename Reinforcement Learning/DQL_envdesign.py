import random
import pygame
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

#window features
WIDTH = 480
HEIGHT = 640
FPS = 60

#colors
BLACK = (0,0,0) #RGB
WHITE = (255,255,255)
RED = (255,0,0)


class Player(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((20,20))
        self.image.fill(BLACK)
        self.rect = self.image.get_rect()
        self.rect.centerx = WIDTH/2
        self.rect.bottom = HEIGHT - 2 
        self.speedX = 0
        
    def update(self, action):
        self.speedX = 0
        key = pygame.key.get_pressed()
        
        if key[pygame.K_LEFT] or action == 0:
            self.speedX = -10
        elif key[pygame.K_RIGHT] or action == 1:
            self.speedX = 10
            
        self.rect.x += self.speedX
        
        if self.rect.right > WIDTH:
            self.rect.right = WIDTH
        if self.rect.left < 0:
            self.rect.left = 0
        
        
    def getCoordinates(self):
        return (self.rect.x , self.rect.y)
         

class Enemy(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((10,10))
        self.image.fill(RED)
        self.rect = self.image.get_rect()
        self.rect.x = random.randrange(0,WIDTH-self.rect.width)
        self.rect.y = random.randrange(0,int(HEIGHT/3))
        self.speedY = 5
        
    def update(self):
        self.rect.y += self.speedY
        
        if self.rect.bottom > HEIGHT:
            self.rect.x = random.randrange(0,WIDTH-self.rect.width)
            self.rect.y = random.randrange(0,int(HEIGHT/3))
            self.speedY = 5
         
        self.speedY += 0.5
            
    def getCoordinates(self):
        return (self.rect.x , self.rect.y)
    
    
class Agent(object):
    def __init__(self):
        #hyperparameters and parameters
        self.stateSize = 4
        self.actionSize = 3
        self.gamma = 0.95
        self.learningRate = 0.001
        self.epsilon = 1
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.memory = deque(maxlen = 1000)
        self.model = self.buildModel()
        
    def buildModel(self):
        #neural network 
        model = Sequential()
        model.add(Dense(48, input_dim=self.stateSize, activation="relu"))
        model.add(Dense(units=self.actionSize, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(lr=self.learningRate))
        return model
    
    def memorize(self, state, action, reward, nextState, done):
        #storage
        self.memory.append((state, action, reward, nextState, done))
    
    def act(self, state):
        #acting, explore or exploit
        state = np.array(state)
        if np.random.rand() < self.epsilon:
            return random.randrange(self.actionSize)
        else:
            actValues = self.model.predict(state)
            return np.argmax(actValues[0])
    
    def replay(self, batchSize):
        #trainin
        if len(self.memory) < batchSize:
            return
        else:
            minibatch = random.sample(self.memory, batchSize)
            for state, action, reward, nextState, done in minibatch:
                state = np.array(state)
                nextState = np.array(nextState)
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
            
 
class Env(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.sprite_group = pygame.sprite.Group()
        self.enemy = pygame.sprite.Group()
        self.player1 = Player()
        self.e1 = Enemy()
        self.e2 = Enemy()
        self.sprite_group.add(self.player1)
        self.sprite_group.add(self.e1)
        self.sprite_group.add(self.e2)
        self.enemy.add(self.e1)
        self.enemy.add(self.e2)
        
        self.reward = 0
        self.done = False
        self.totalRew = 0
        self.agent = Agent()
        
    def findDistance(self, a, b):
        
        return a - b
    
    def step(self, action):
        stateList = []
        
        self.player1.update(action)
        self.enemy.update()
        
        next_player1State = self.player1.getCoordinates()
        next_e1State = self.e1.getCoordinates()
        next_e2State = self.e2.getCoordinates()
        
        stateList.append(self.findDistance(next_player1State[0], next_e1State[0]))
        stateList.append(self.findDistance(next_player1State[1], next_e1State[1]))
        stateList.append(self.findDistance(next_player1State[0], next_e2State[0]))
        stateList.append(self.findDistance(next_player1State[1], next_e2State[1]))
        
        return [stateList]
    
    def reset(self):
        self.sprite_group = pygame.sprite.Group()
        self.enemy = pygame.sprite.Group()
        self.player1 = Player()
        self.e1 = Enemy()
        self.e2 = Enemy()
        self.sprite_group.add(self.player1)
        self.sprite_group.add(self.e1)
        self.sprite_group.add(self.e2)
        self.enemy.add(self.e1)
        self.enemy.add(self.e2)        

        self.reward = 0
        self.done = False
        self.totalRew = 0
        
        stateList = []

        player1State = self.player1.getCoordinates()
        e1State = self.e1.getCoordinates()
        e2State = self.e2.getCoordinates()

        stateList.append(self.findDistance(player1State[0], e1State[0]))
        stateList.append(self.findDistance(player1State[1], e1State[1]))
        stateList.append(self.findDistance(player1State[0], e2State[0]))
        stateList.append(self.findDistance(player1State[1], e2State[1]))    
        
        return [stateList]
    
    def run(self):
        #Game loop
        state = self.reset()
        running = True
        batch_size = 16
        while running:
            self.reward = 2
            clock.tick(FPS)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            action = self.agent.act(state)
            nextState = self.step(action)
            self.totalRew += self.reward

            hit = pygame.sprite.spritecollide(self.player1,self.enemy, False, pygame.sprite.collide_rect)
            if hit:
                self.reward = -2000
                self.totalRew += self.reward
                self.done = True
                running = False
                print("Total reward: {}".format(self.totalRew))
            
            self.agent.memorize(state, action, self.reward, nextState, self.done)
            state = nextState
            self.agent.replay(batch_size)
            
            screen.fill(WHITE)        
            self.sprite_group.draw(screen)
    
            pygame.display.flip()
        
        self.agent.adaptiveEpsilon()
        pygame.quit()

if __name__ == "__main__":
    env = Env()
    liste = []
    t = 0
    while True:
        t += 1
        print("Episode: {}".format(t))
        liste.append(env.totalRew)
        
        pygame.init()
        screen = pygame.display.set_mode((WIDTH,HEIGHT))
        pygame.display.set_caption("Welcome to the Game")
        clock = pygame.time.Clock()
        
        env.run()




