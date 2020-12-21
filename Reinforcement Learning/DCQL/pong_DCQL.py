import pygame
import random

FPS = 60

WINDOW_WIDTH = 400
WINDOW_HEIGHT = 420
GAME_HEIGHT = 400

PADDLE_WIDTH = 15
PADDLE_HEIGHT = 60
PADDLE_BUFFER = 15

BALL_WIDTH = 20
BALL_HEIGHT = 20

PADDLE_SPEED = 3
BALL_SPEEDX = 2
BALL_SPEEDY = 2

WHITE = (255,255,255)
BLACK = (0,0,0)

screen = pygame.display.set_mode((WINDOW_WIDTH,WINDOW_HEIGHT))

class PongGame:
    
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Pong DCQL")
        
        self.paddle1Ypos = GAME_HEIGHT/2 - PADDLE_HEIGHT/2
        self.paddle2Ypos = GAME_HEIGHT/2 - PADDLE_HEIGHT/2
        
        self.clock = pygame.time.Clock()
        self.gScore = 0.0
        
        self.ballXdirection = random.sample([-1,1],1)[0]
        self.ballYdirection = random.sample([-1,1],1)[0]
        self.ballXpos = WINDOW_WIDTH/2
        self.ballYpos = random.randint(0,9)*(WINDOW_HEIGHT-BALL_HEIGHT)/9
        
    def drawPaddle(self, switch):
        if switch == "left":
            paddle = pygame.Rect(PADDLE_BUFFER, self.paddle1Ypos, PADDLE_WIDTH, PADDLE_HEIGHT)
        else:
            paddle = pygame.Rect(WINDOW_WIDTH-PADDLE_BUFFER-PADDLE_WIDTH, self.paddle2Ypos, PADDLE_WIDTH, PADDLE_HEIGHT)
        pygame.draw.rect(screen, WHITE, paddle)
        
    def drawBall(self):
        ball = pygame.Rect(self.ballXpos, self.ballYpos, BALL_WIDTH, BALL_HEIGHT)
        pygame.draw.rect(screen, WHITE, ball)
        
    def updatePaddle(self, switch, action):
        dft = 7.5
        
        if switch == "left":
            if action == 1:
                self.paddle1Ypos = self.paddle1Ypos - PADDLE_SPEED*dft #upward
            elif action == 2:
                self.paddle1Ypos = self.paddle1Ypos + PADDLE_SPEED*dft #downward
                
            if self.paddle1Ypos < 0:
                self.paddle1Ypos = 0
            if self.paddle1Ypos > GAME_HEIGHT - PADDLE_HEIGHT:
                self.paddle1Ypos = 0
        else:
            if self.paddle2Ypos + PADDLE_HEIGHT/2 < self.ballYpos + BALL_HEIGHT/2:
                self.paddle2Ypos = self.paddle2Ypos + PADDLE_SPEED*dft
            elif self.paddle2Ypos + PADDLE_HEIGHT/2 > self.ballYpos + BALL_HEIGHT/2:
                self.paddle2Ypos = self.paddle2Ypos - PADDLE_SPEED*dft
            
            if self.paddle2Ypos < 0:
                self.paddle2Ypos = 0
            if self.paddle2Ypos > GAME_HEIGHT - PADDLE_HEIGHT:
                self.paddle2Ypos = 0        
    
    def updateBall(self):
        dft = 7.5
        score = -0.05
        
        self.ballXpos = self.ballXpos + self.ballXdirection*BALL_SPEEDX*dft
        self.ballYpos = self.ballYpos + self.ballYdirection*BALL_SPEEDY*dft
        
        if (self.ballXpos <= (PADDLE_BUFFER + PADDLE_WIDTH)) and ((self.ballYpos + BALL_HEIGHT) >= self.paddle1Ypos) and (self.ballYpos <= (self.paddle1Ypos + PADDLE_HEIGHT)) and (self.ballXdirection == -1):
            self.ballXdirection = 1
            score = 10
        elif self.ballXpos <= 0:
            self.ballXdirection = 1
            score = -10
            
        if (self.ballXpos >= (WINDOW_WIDTH - PADDLE_WIDTH - PADDLE_BUFFER)) and ((self.ballYpos + BALL_HEIGHT) >= self.paddle2Ypos) and (self.ballYpos <= (self.paddle2Ypos + PADDLE_HEIGHT)) and (self.ballXdirection == 1):
            self.ballXdirection = -1
        elif self.ballXpos >= WINDOW_WIDTH - BALL_WIDTH:
            self.ballXdirection = -1
            
        if self.ballYpos <= 0:
            self.ballYpos = 0
            self.ballYdirection = 1
        elif self.ballYpos >= GAME_HEIGHT - BALL_HEIGHT:
            self.ballYpos = GAME_HEIGHT - BALL_HEIGHT
            self.ballYdirection = -1
            
        return [score]
        
        
    def initialDisplay(self):
        pygame.event.pump()
        screen.fill(BLACK)
        
        self.drawPaddle("left")
        self.drawPaddle("right")
        
        self.drawBall()
        
        pygame.display.flip()
        
    
    def playNextMove(self, action):
        self.deltaFrameTime = self.clock.tick(FPS)
        pygame.event.pump()
        score = 0
        screen.fill(BLACK)
        
        self.updatePaddle("left", action)
        self.drawPaddle("left")
        
        self.updatePaddle("right", action)
        self.drawPaddle("right")
        
        [score] = self.updateBall()
        
        self.drawBall()
        
        if (score > 0.5) or(score < 0.5):
            self.gScore = 0.9*self.gScore + 0.1*score
                
        screenImage = pygame.surfarray.array3d(pygame.display.get_surface())
        
        pygame.display.flip()
        
        return [score, screenImage]