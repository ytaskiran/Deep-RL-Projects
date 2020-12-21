import pong_DCQL
import agent_DCQL
import numpy as np
import skimage
import warnings
warnings.filterwarnings("ignore")

TOTAL_TRAINTIME = 100000
IMGHEIGHT = 40
IMGWIDTH = 40
IMGHISTORY = 4


def processGameImage(rawImage):
    grayImage = skimage.color.rgb2gray(rawImage)
    croppedImage = grayImage[0:400,0:400]
    reducedImage = skimage.transform.resize(croppedImage,(IMGHEIGHT, IMGWIDTH))
    reducedImage = skimage.exposure.rescale_intensity(reducedImage, out_range=(0,255))
    reducedImage = reducedImage / 128
    return reducedImage
    
def trainExperiment():
    trainHistory = []
    theGame = pong_DCQL.PongGame()
    theGame.initialDisplay()
    
    theAgent = agent_DCQL.Agent()
    bestAction = 0
    [initialScore, initialScreenImage] = theGame.playNextMove(bestAction)
    initialGameImage = processGameImage(initialScreenImage)
    gameState = np.stack((initialGameImage,initialGameImage,initialGameImage,initialGameImage), axis=2)
    gameState = gameState.reshape(1, gameState.shape[0], gameState.shape[1], gameState.shape[2])
    
    for i in range(TOTAL_TRAINTIME):                   
        bestAction = theAgent.findBestAct(gameState)
        [returnScore, newScreenImage] = theGame.playNextMove(bestAction)
        
        newGameImage = processGameImage(newScreenImage)
        newGameImage = newGameImage.reshape(1, newGameImage.shape[0],newGameImage.shape[1],1)
        nextState = np.append(newGameImage, gameState[:,:,:,:3], axis = 3)
        theAgent.captureSample((gameState,bestAction,returnScore,nextState))
        theAgent.process()
        gameState = nextState
                
        if i%250 == 0:
            print("Train time: ",i, "game score: ", theGame.gScore)
            trainHistory.append(theGame.gScore)
            
trainExperiment()

















