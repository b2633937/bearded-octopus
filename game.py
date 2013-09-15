import pygame, sys
import random
import pickle 
import numpy as np
import matplotlib.pyplot as plt
from pygame.locals import *
from players import *
from locals import *

class Game(object, ):

    def __init__(self):
        self.boardSize = (11, 11)
        self.agents = [] 
        self.fixedInitPositions = [(0,0), (5,5)]
        reInitmode = 'fixed' # 'random' #
        assignment = 1
        verbose = 1
        draw = 0
        episodes = 2  
        rnds = 2 #100 


        if draw:
            #set up the window
            pygame.init()
            fpsClock = pygame.time.Clock()
            self.screen = GameScreen(pygame.display.set_mode((800, 600), 0, 32), self.boardSize)
       
        # instantiate agents
        player1 = Human(self)
        player2 = ValueIteration(self) #RandomComputer(self)#

        self.agents.append(Agent(player = player1, role='predator', nr=len(self.agents), img=IMAGES['boy']))
        self.agents.append(Agent(player = player2, role='prey', nr=len(self.agents), img=IMAGES['princess']))
        self.initPositions(reInitmode)


        episode = 0
        rnd = 0
        stats = np.zeros((episodes, rnds))
        activeAgent = 0
        turn = 0
        caught = 0

        while True: #the main game loop
            if draw:
                self.screen.draw(turn, caught, self.agents)
                self.screen.handleUserInput() #listen for Quit events etc.

            if self.gameEnds(): #check for game end
                SOUNDS['caught'].play()
                self.initPositions(reInitmode)
                stats[episode, rnd] = turn
                caught += 1
                turn = 0
                activeAgent = 0
                rnd += 1
                if rnd == rnds:
                    for agent in self.agents:
                        agent.quit()
                    episode += 1
                    rnd = 0
                    turn = 0
                    caught = 0
                    if episode == episodes:
                        pygame.quit()
                        self.processResults(stats, rnds, episodes)
                        sys.exit()
            else:
                #Agent takes move 
                observation = self.getObservation(activeAgent, observability = 'fo')
                action = self.agents[activeAgent].getAction(observation)

                if action != None: 
                    self.agents[activeAgent].POS = self.transition(self.getState(), action, activeAgent)
                    observation = self.getObservation(activeAgent, observability = 'fo')
                    reward = self.getReward(self.agents[activeAgent].role)
                    self.agents[activeAgent].finalize(reward, observation)
                    #pass turn to next agent
                    if activeAgent == len(self.agents)-1:
                        activeAgent = 0
                        turn += 1
                        if assignment == 1 and verbose:
                            print [agent.POS for agent in self.agents]
                    else:
                        activeAgent += 1
         
            fpsClock.tick(30)

    def getObservation(self, agentNr, observability):
        #positions also contains own position!
        if observability == 'fo': #full obervability
            return Observation(self.getState())
        else: 
            sys.exit('ERROR: unknown observability parameter')

    def getState(self):
        return [agent.POS for agent in self.agents]

    def transition(self, state, action, agentNr):
        return ((self.agents[agentNr].POS[0]+EFFECTS[action][0])%self.boardSize[0], 
            (self.agents[agentNr].POS[1]+EFFECTS[action][1])%self.boardSize[1])

    def gameEnds(self): #game ends if any 2 agents share same position
        s = set()
        for agent in self.agents:
            s.add(agent.POS)
        return len(s) != len(self.agents)

    def getReward(self, role): 
        nroPreds = 0
        preds = set()
        s = set()
        reward = 0
        for agent in self.agents:
            if agent.role == 'predator':
                nroPreds += 1
                preds.add(agent.POS)
            s.add(agent.POS)
        if len(preds) != nroPreds:
            #two predators share position
            reward = -10
        elif len(s) != len(self.agents):
            #a predator shares position with prey
            reward = 10 
        if role == 'prey':
            #reverse reward
            reward *= -1
            if reward > 0: # prey gets no reward if 2 predators collide
                reward = 0
        return reward

    def initPositions(self, mode):
        if mode == 'random': #still prevent agents from getting same initial position
            positions = set()
            for agent in self.agents:
                position = (random.randint(0,self.boardSize[0]-1), random.randint(0,self.boardSize[1]-1))
                while position in positions: 
                    position = (random.randint(0,self.boardSize[0]-1), random.randint(0,self.boardSize[1]-1))
                agent.POS = position
        if mode == 'fixed':
            for i in xrange(len(self.agents)):
                self.agents[i].POS = self.fixedInitPositions[i]

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
    def __init__(self, displaySurf, boardSize):
        super(GameScreen, self).__init__()
        self.displaySurf = displaySurf
        self.boardSize = boardSize
        self.boardOffsetX = 250
        self.boardoffsetY = 50
        self.tileSize = min(500/self.boardSize[0], 500/self.boardSize[1])
        self.boardDim = (self.boardSize[0]*self.tileSize, self.boardSize[1]*self.tileSize)
        IMAGES['backgroundImg'] = pygame.transform.scale(IMAGES['backgroundImg'],
            (self.boardDim[1], self.boardDim[0]))
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
        pygame.draw.rect(self.displaySurf, RED, (self.boardOffsetX, self.boardoffsetY, self.boardDim[1], self.boardDim[0]), 3) #draw border
        for i in xrange(1, self.boardSize[0]): #draw horizontal lines
            pygame.draw.line(self.displaySurf, RED, (self.boardOffsetX, self.boardoffsetY+i*self.tileSize), 
                (self.boardOffsetX+self.boardDim[1], self.boardoffsetY+i*self.tileSize),1)
        for i in xrange(1, self.boardSize[1]): #draw vertical lines
            pygame.draw.line(self.displaySurf, RED, (self.boardOffsetX+i*self.tileSize, 50), 
                (self.boardOffsetX+i*self.tileSize, self.boardoffsetY+self.boardDim[0]),1)
        self.displaySurf.blit(self.displaySurf, (0, 0))
   
    def draw2Tile(self, (y, x), img):
        screenX = (x*self.tileSize + 0.5*self.tileSize + self.boardOffsetX)
        screenY = (y*self.tileSize + 0.5*self.tileSize + self.boardoffsetY)
        screenY += 9 #TODO: handle picture ofsett
        self.displaySurf.blit(img, img.get_rect(center=(screenX,screenY)).topleft)

    def draw(self, turn, caught, agents):
        self.displaySurf.fill(WHITE)
        self.displaySurf.blit(IMAGES['backgroundImg'], (self.boardOffsetX, self.boardoffsetY))
        self.drawBoard() 
        for agent in agents:
            self.draw2Tile(agent.POS, agent.img)
        self.displaySurf.blit(self.quitTextSurf, self.quitTextRect)
        self.textSurfaceObj = self.fontObj.render('Turn: ' + str(turn), True, BLUE)
        self.displaySurf.blit(self.textSurfaceObj, self.textRectObj)
        self.caughtTextSurf = self.fontObj.render('Caught: ' + str(caught), True, BLUE)
        self.displaySurf.blit(self.caughtTextSurf, self.caughtTextRect)
        pygame.display.update()
