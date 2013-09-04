import pygame, sys
import random
import pickle 
import numpy as np
import matplotlib.pyplot as plt
import time #probably obsolete
from pygame.locals import *
from players import *
from locals import *

"""
TODO: 

B - evaluate state after oponent took action > Qlearning!
B - use mirroring to reduce state space
B - pretty print state space 
B - pretty print avg graph with std     - DONE            
B - create visual switch
B - fix 'QUIT'-button
B - commented TODO's

T - implement policy evaluation
E - implement policy iteration
H - implement value iteration

"""


def main():
    global DISPLAYSURF, AGENTS, BOARDSIZE, TILESIZE, BOARDOFFSETX, BOARDOFFSETY #TODO: get rid of globals
    pygame.init()
    assignment = 1
    verbose = 0
    draw = 1
    windowWidth = 800
    windowheight = 600
    BOARDOFFSETX = 250
    BOARDOFFSETY = 50
    BOARDSIZE = 11 # always uneven in order for state to work! TODO: fix!
    TILESIZE = 45
    FPS = 30 #frames per second setting
    reInitmode = 'fixed' # 'random' #
    AGENTS = [] 
    fpsClock = pygame.time.Clock()
    loadSounds()
    loadImages()

    if draw:
        #set up the window
        DISPLAYSURF = pygame.display.set_mode((windowWidth, windowheight), 0, 32)
        gameScreen = GameScreen(DISPLAYSURF)
   
    # instantiate agents
    player1 = Player.new('RandomComputer') #'Human') #
    player2 = Player.new('RandomComputer')

    AGENTS.append(Agent(player = player1, role='predator', nr=len(AGENTS), img=IMAGES['boy']))
    AGENTS.append(Agent(player = player2, role='prey', nr=len(AGENTS), img=IMAGES['princess']))

    initPositions(AGENTS, reInitmode)

    episodes = 2 
    episode = 0
    rnds = 2 #100 
    rnd = 0
    stats = np.zeros((episodes, rnds))
  
    activeAgent = 0
    turn = 0
    caught = 0

    while True: #the main game loop
        if draw:
            gameScreen.draw(turn, caught)
        handleUserInput() #listen for Quit events etc.
        
        if gameEnds(): #check for game end
            GAMESOUNDS['caught'].play()
            initPositions(AGENTS, reInitmode)
            stats[episode, rnd] = turn
            caught += 1
            turn = 0
            activeAgent = 0
            rnd += 1
            if rnd == rnds:
                for agent in AGENTS:
                    agent.quit()
                episode += 1
                rnd = 0
                turn = 0
                caught = 0
                if episode == episodes:
                    pygame.quit()
                    processResults(stats, rnds, episodes)
                    sys.exit()
        else:
            #Agent takes move 
            observation = getObservation(activeAgent, observability = 'fo')
            action = AGENTS[activeAgent].getAction(observation)

            if action != None: 
                transit(activeAgent, action)
                observation = getObservation(activeAgent, observability = 'fo')
                reward = getReward(AGENTS[activeAgent].role)
                AGENTS[activeAgent].finalize(reward, observation)
                #pass turn to next agent
                if activeAgent == len(AGENTS)-1:
                    activeAgent = 0
                    turn += 1
                    if assignment == 1 and verbose:
                        print [agent.POS for agent in AGENTS]
                else:
                    activeAgent += 1
     
        fpsClock.tick(FPS)

def getObservation(agentNr, observability):
    #positions also contains own position!
    if observability == 'fo': #full obervability
        positions = [agent.POS for agent in AGENTS]
    return Observation(positions)

def transit(activeAgent, action):
    AGENTS[activeAgent].POS = action2Tile(action, AGENTS[activeAgent].POS)

def gameEnds(): #game ends if any 2 agents share same position
    s = set()
    for agent in AGENTS:
        s.add(agent.POS)
    return len(s) != len(AGENTS)

def getReward(role): 
    nroPreds = 0
    preds = set()
    s = set()
    reward = 0
    for agent in AGENTS:
        if agent.role == 'predator':
            nroPreds += 1
            preds.add(agent.POS)
        s.add(agent.POS)
    if len(preds) != nroPreds:
        #two predators share position
        reward = -10
    elif len(s) != len(AGENTS):
        #a predator shares position with prey
        reward = 10 
    if role == 'prey':
        #reverse reward
        reward *= -1
        if reward > 0: # prey gets no reward if 2 predators collide
            reward = 0
    return reward

def initPositions(AGENTS, mode):
    if mode == 'random': #still prevent agents from getting same initial position
        positions = set()
        for agent in AGENTS:
            position = (random.randint(0,9), random.randint(0,9))
            while position in positions: 
                position = (random.randint(0,9), random.randint(0,9))
            agent.POS = position
    fixedInitPositions = [(0,0), (5,5)] #TODO: move to Player?
    if mode == 'fixed':
        for i in xrange(len(AGENTS)):
            AGENTS[i].POS = fixedInitPositions[i]

def handleUserInput():
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

        if event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                pygame.quit()
                sys.exit()

        if event.type == MOUSEBUTTONUP:
            mousex, mousey = event.pos
            if quitTextRect.collidepoint(mousex, mousey):
                pygame.quit()
                sys.exit()
        pygame.event.post(event)

def draw2Tile((x, y), img):
    screenX = (x*TILESIZE + 0.5*TILESIZE + BOARDOFFSETX)
    screenY = (y*TILESIZE + 0.5*TILESIZE + BOARDOFFSETY)
    screenY += 9 #TODO: handle picture size ofsett
    DISPLAYSURF.blit(img, img.get_rect(center=(screenX,screenY)).topleft)

def drawBoard():
    pygame.draw.rect(DISPLAYSURF, RED, (BOARDOFFSETX, 50, 495, 495), 3)
    for i in xrange(1, BOARDSIZE):
        pygame.draw.line(DISPLAYSURF, RED, (BOARDOFFSETX, 50+i*(495/BOARDSIZE)), (BOARDOFFSETX+495, 50+i*(495/BOARDSIZE)),1)
        pygame.draw.line(DISPLAYSURF, RED, (BOARDOFFSETX+i*(495/BOARDSIZE), 50), (BOARDOFFSETX+i*(495/BOARDSIZE), 50+495),1)
    DISPLAYSURF.blit(DISPLAYSURF, (0, 0))

def loadSounds():
    global GAMESOUNDS
    GAMESOUNDS = {}
    GAMESOUNDS['caught'] = pygame.mixer.Sound('match4.wav')

def loadImages():
    global IMAGES
    IMAGES = {}
    #load and scale background image
    backgroundImg = pygame.image.load('flippyboard.png')
    backgroundImg = pygame.transform.scale(backgroundImg, (495,495))
    IMAGES['backgroundImg'] =  backgroundImg 
    #load agent images
    IMAGES['boy'] = pygame.image.load('princess.png')
    IMAGES['princess'] = pygame.image.load('boy.png')

def processResults(stats, rnds, episodes):
    try:
        pickle.dump(stats, open('stats', "wb"), pickle.HIGHEST_PROTOCOL)    
    except:
        print "can't write stats file"

    plt.figure('Results')
    x = np.arange(1,rnds+1,1)
    y = stats.sum(0) / float(episodes)
    std = stats.std(0)
    print x.shape
    print y.shape
    errorfill(x, y, std)
    plt.show()



#####################################################################################

class Observation(object):
    """docstring for Observation"""
    def __init__(self, positions):
        super(Observation, self).__init__()
        self.positions = positions

class GameScreen(object):
    """docstring for GameScreen"""
    def __init__(self, DISPLAYSURF):
        super(GameScreen, self).__init__()
        self.DISPLAYSURF = DISPLAYSURF
        pygame.display.set_caption('Kiss of Death')
        self.fontObj = pygame.font.Font('freesansbold.ttf', 32)

        self.textSurfaceObj = self.fontObj.render('Turn: 00', True, BLUE)
        self.textRectObj = self.textSurfaceObj.get_rect()
        self.textRectObj.topleft = (20, 50)

        self.caughtTextSurf = self.fontObj.render('Caught: 00', True, BLUE)
        self.caughtTextRect = self.caughtTextSurf.get_rect()
        self.caughtTextRect.topleft = (20, 100)

        self.quitTextSurf = self.fontObj.render('Quit', True, BLUE)
        self.quitTextRect = self.quitTextSurf.get_rect()
        self.quitTextRect.topleft = (20, 495)

    def draw(self, turn, caught):
        DISPLAYSURF.fill(WHITE)
        DISPLAYSURF.blit(IMAGES['backgroundImg'], (BOARDOFFSETX, 50))
        drawBoard() 
        for agent in AGENTS:
            draw2Tile(agent.POS, agent.img)
        DISPLAYSURF.blit(self.quitTextSurf, self.quitTextRect)
        self.textSurfaceObj = self.fontObj.render('Turn: ' + str(turn), True, BLUE)
        DISPLAYSURF.blit(self.textSurfaceObj, self.textRectObj)
        self.caughtTextSurf = self.fontObj.render('Caught: ' + str(caught), True, BLUE)
        DISPLAYSURF.blit(self.caughtTextSurf, self.caughtTextRect)
        pygame.display.update()

        
#####################################################################################

if __name__ == '__main__':
    main()


