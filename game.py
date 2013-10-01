import pygame, sys, random, pickle, numpy as np, matplotlib.pyplot as plt
from pygame.locals import *
from players import *
from locals import *

"""Central File which runs a game instance called through assignment_x.py"""

class Game(object):

    def __init__(self, boardSize, verbose, draw, rounds, episodes):
        self.settings = type('Settings', (object,), dict(
            boardSize = boardSize,
            verbose = verbose,
            draw = draw,
            rounds = rounds,
            episodes = episodes,
            fixedInitPositions = [],
            playerImages = [],
            playerRoles = [],
            nroPlayers = 0
            ))
        self.players = []


    def play(self):
        settings = self.settings
        state = State()
        stats = np.zeros((settings.rounds, settings.episodes))

        pygame.init()
        if settings.draw:
            #set up the window
            fpsClock = pygame.time.Clock()
            self.screen = GameScreen(pygame.display.set_mode((800, 600), 0, 32), settings.boardSize)
       
        self.initPositions(settings, state)
        activePlayerNr = 0
        #Main Game Loop
        while True: 
            if settings.draw:
                self.screen.draw(settings, state, self.players)
                self.screen.handleUserInput() #listen for Quit events etc.

            if self.gameEnds(state): #check for game end
                SOUNDS['caught'].play()
                self.initPositions(settings, state)
                stats[state.rnd, state.episode] = state.turn
                state.caught += 1 #TODO: check if episode ended caused by duplicate pred pos
                state.turn = 0
                self.activePlayerNr = 0
                state.episode += 1
                if state.episode == settings.episodes:
                    for player in self.players:
                        player.quit() #allows saving files etc.
                    state.caught = 0
                    state.rnd += 1
                    if rnd == settings.rounds:
                        pygame.quit()
                        return self.processResults(stats, settings.episodes, settings.rounds)
            else:
                #Agent has to take move 
                self.players[activePlayerNr].observe(self.getObservation(settings, state, activePlayerNr, observability = 'fo'))
                action = self.players[activePlayerNr].getAction()
                if action != None: 
                    self.stateTransition(settings, state, action, activePlayerNr)
                    self.players[activePlayerNr].observe(self.getObservation(settings, state, activePlayerNr, observability = 'fo'))
                    reward = self.getReward(settings, state, action)
                    self.players[activePlayerNr].finalize(reward)
                    #pass turn to next player
                    if activePlayerNr == len(self.players)-1:
                        activePlayerNr = 0
                        state.turn += 1
                        if settings.verbose:
                            print state.positions
                    else:
                        activePlayerNr += 1
            if settings.draw:
                fpsClock.tick(30)

    def addPlayer(self, player, role, fixedInitPos, img):
        player.nr = len(self.players) 
        self.players.append(player)
        self.settings.fixedInitPositions.append(fixedInitPos)
        self.settings.playerImages.append(img)
        self.settings.playerRoles.append(role)
        self.settings.nroPlayers += 1

    def getObservation(self, settings, state, playerNr, observability):
        if observability == 'fo': #full obervability
            return Observation(settings, state, playerNr)
        else: 
            sys.exit('ERROR: unknown observability parameter')

    def stateTransition(self, settings, state, action, playerNr):
        state.positions[playerNr] = ((state.positions[playerNr][0]+EFFECTS[action][0])%settings.boardSize[0], 
            (state.positions[playerNr][1]+EFFECTS[action][1])%settings.boardSize[1])

    def gameEnds(self, state): #game ends if any 2 players share same position
        return len(set(state.positions)) != len(self.players)

    def getReward(self, settings, state, action): 
        nroPreds = 0
        preds = set()
        s = set()
        reward = 0
        for i in xrange(settings.nroPlayers):
        # for player in self.players:
            if settings.playerRoles[i] == PREDATOR:
                nroPreds += 1
                preds.add(state.positions[i])
            s.add(state.positions[i])
        if len(preds) != nroPreds:
            #two predators share position
            reward = -10
        elif len(s) != len(self.players):
            #a predator shares position with prey
            reward = 10 
        if settings.playerRoles[i] == PREY:
            #reverse reward
            reward *= -1
            if reward > 0: # prey gets no reward if 2 predators collide
                reward = 0
        return reward

    def initPositions(self, settings, state):
        #give all fixedPos player their position
        positionSet = set()
        positions = [None]*len(self.players)
        for i in xrange(len(self.players)):
            if settings.fixedInitPositions[i] != None:
                positions[i] = settings.fixedInitPositions[i]
                positionSet.add(settings.fixedInitPositions[i])
        #now give all None fixedPos players a position
        for i in xrange(len(positions)):
            if positions[i] == None:
                while True: 
                    position = (random.randint(0,settings.boardSize[0]-1), random.randint(0,settings.boardSize[1]-1))
                    if position not in positionSet:
                        break
                positions[i] = position
        state.positions = positions

    def processResults(self, stats, episodes, rnds):
        try:
            pickle.dump(stats, open('stats', "wb"), pickle.HIGHEST_PROTOCOL)    
        except:
            print "can't write stats file"
        print stats
        avg = stats.sum(1) / float(episodes)
        std = stats.std(1)
        print 'average of: ', avg
        print 'standard deviation of: ', std


#####################################################################################

class Observation(object):
    """docstring for Observation"""
    def __init__(self, settings, state, playerNr):
        super(Observation, self).__init__()
        self.playerNr = playerNr
        self.roles = settings.playerRoles
        self.positions = state.positions
        self.boardSize = settings.boardSize


class State(object):
    """docstring for State"""
    def __init__(self):
        super(State, self).__init__()
        self.positions = None
        self.rnd = 0
        self.episode = 0
        self.turn = 0
        self.caught = 0
        self.activePlayerNr = 0


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

    def draw(self, settings, state, players):
        self.displaySurf.fill(WHITE)
        self.displaySurf.blit(IMAGES['backgroundImg'], (self.boardOffsetX, self.boardoffsetY))
        self.drawBoard() 
        for i in xrange(len(players)):
            self.draw2Tile(state.positions[i], settings.playerImages[i])
        self.displaySurf.blit(self.quitTextSurf, self.quitTextRect)
        self.textSurfaceObj = self.fontObj.render('Turn: ' + str(state.turn), True, BLUE)
        self.displaySurf.blit(self.textSurfaceObj, self.textRectObj)
        self.caughtTextSurf = self.fontObj.render('Caught: ' + str(state.caught), True, BLUE)
        self.displaySurf.blit(self.caughtTextSurf, self.caughtTextRect)
        pygame.display.update()
