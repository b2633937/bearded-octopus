import pygame, sys
import random
import pickle 
import numpy as np
import matplotlib.pyplot as plt
from pygame.locals import *
from players import *
from locals import *

class Game(object):

    def __init__(self):

        global SCREEN, AGENTS
        pygame.init()
        assignment = 1
        verbose = 0
        draw = 1
        windowWidth = 800
        windowheight = 600
        boardOffsetX = 250
        boardoffsetY = 50
        boardSize = 10 # always uneven in order for state to work! TODO: fix!
        tileSize = 45
        FPS = 30 #frames per second setting
        reInitmode = 'fixed' # 'random' #
        AGENTS = [] 
        fpsClock = pygame.time.Clock()


        if draw:
            #set up the window
            SCREEN = GameScreen(pygame.display.set_mode((windowWidth, windowheight), 0, 32), 
                boardSize, boardOffsetX, boardoffsetY, tileSize)
       
        # instantiate agents
        player1 = Player.new('Human') #'RandomComputer') #
        player2 = Player.new('RandomComputer')

  
    def getObservation(self, agentNr, observability):
        #positions also contains own position!
        if observability == 'fo': #full obervability
            positions = [agent.POS for agent in AGENTS]
        return Observation(positions)

    def transit(self, activeAgent, action):
        AGENTS[activeAgent].POS = action2Tile(action, AGENTS[activeAgent].POS)

    def gameEnds(self): #game ends if any 2 agents share same position
        s = set()
        for agent in AGENTS:
            s.add(agent.POS)
        return len(s) != len(AGENTS)

    def getReward(self, role): 
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

    def initPositions(self, AGENTS, mode):
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

    def processResults(self, stats, rnds, episodes):
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
    def __init__(self, DISPLAYSURF, boardSize, boardOffsetX, boardoffsetY, tileSize):
        super(GameScreen, self).__init__()
        self.DISPLAYSURF = DISPLAYSURF
        self.boardSize = boardSize
        self.boardOffsetX = boardOffsetX
        self.boardoffsetY = boardoffsetY
        self.tileSize = tileSize
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

    def handleUserInput(self):
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

                if self.quitTextRect.collidepoint(mousex, mousey):
                    pygame.quit()
                    sys.exit()
            pygame.event.post(event)

    def drawBoard(self):
        pygame.draw.rect(self.DISPLAYSURF, RED, (self.boardOffsetX, 50, 495, 495), 3)
        for i in xrange(1, self.boardSize):
            pygame.draw.line(self.DISPLAYSURF, RED, (self.boardOffsetX, 50+i*(495/self.boardSize)), (self.boardOffsetX+495, 50+i*(495/self.boardSize)),1)
            pygame.draw.line(self.DISPLAYSURF, RED, (self.boardOffsetX+i*(495/self.boardSize), 50), (self.boardOffsetX+i*(495/self.boardSize), 50+495),1)
        self.DISPLAYSURF.blit(self.DISPLAYSURF, (0, 0))
   
    def draw2Tile(self, (x, y), img):
        screenX = (x*self.tileSize + 0.5*self.tileSize + self.boardOffsetX)
        screenY = (y*self.tileSize + 0.5*self.tileSize + self.boardoffsetY)
        screenY += 9 #TODO: handle picture size ofsett
        self.DISPLAYSURF.blit(img, img.get_rect(center=(screenX,screenY)).topleft)

    def draw(self, turn, caught):
        self.DISPLAYSURF.fill(WHITE)
        self.DISPLAYSURF.blit(IMAGES['backgroundImg'], (self.boardOffsetX, 50))
        self.drawBoard() 
        for agent in AGENTS:
            self.draw2Tile(agent.POS, agent.img)
        self.DISPLAYSURF.blit(self.quitTextSurf, self.quitTextRect)
        self.textSurfaceObj = self.fontObj.render('Turn: ' + str(turn), True, BLUE)
        self.DISPLAYSURF.blit(self.textSurfaceObj, self.textRectObj)
        self.caughtTextSurf = self.fontObj.render('Caught: ' + str(caught), True, BLUE)
        self.DISPLAYSURF.blit(self.caughtTextSurf, self.caughtTextRect)
        pygame.display.update()
