import pygame, sys, random, pickle, numpy as np, matplotlib.pyplot as plt
from pygame.locals import *
from agents import *
from locals import *

"""Central File which runs a game instance called through assignment_x.py"""

class Game(object):


    def __init__(self, boardSize, verbose, draw, episodes, maxTurn, simultaniousActions, preyTrip):
        self.settings = type('Settings', (object,), dict(
            boardSize = boardSize,
            verbose = verbose,
            draw = draw,
            episodes = episodes,
            fixedInitPositions = [],
            playerImages = [],
            playerRoles = [],
            nroPlayers = 0,
            maxTurn = maxTurn,
            simultaniousActions = simultaniousActions,
            preyTrip = preyTrip
            ))
        self.players = []


    def play(self):
        settings = self.settings
        state = State()
        turns = np.zeros(settings.episodes)
        outcomes = []
        results = {'turns': turns, 'outcomes': outcomes} 


        pygame.init()
        if settings.draw:
            #set up the window
            fpsClock = pygame.time.Clock()
            self.screen = GameScreen(pygame.display.set_mode((800, 600), 0, 32), settings.boardSize)
       
        self.initState(settings, state)

        simultaniousActions = True
        #------------------------ MAIN GAME LOOP ------------------------
        while True: 
            if settings.draw:
                self.screen.draw(settings, state, self.players)
                self.screen.handleUserInput() #listen for Quit events etc.

            outcome = self.getOutcome(settings, state)
 
            if self.gameEnds(settings, state) or (settings.maxTurn and state.rnd >= settings.maxTurn): #check for game end
                if settings.draw and self.gameEnds(settings, state):
                    SOUNDS['caught'].play() #don't play when episodelength is reached

                for i in xrange(self.settings.nroPlayers):
                    self.players[i].observe(self.getObservation(settings, state, i, observability = 'fo'))
                    self.players[i].finalize(self.getReward(outcome, settings.playerRoles[i]))
                    self.players[i].finalizeEpisode(self.getReward(outcome, settings.playerRoles[i]))

                turns[state.episode] = state.rnd
                outcomes.append(outcome)
                self.initState(settings, state)
                state.caught += 1 #TODO: check if episode ended caused by duplicate pred pos
                state.episode += 1
                if state.episode == settings.episodes:
                    for player in self.players:
                        player.quit() #allows saving files etc.
                    pygame.quit()
                    return {'turns': turns, 'outcomes': outcomes}
            else:
                #observe
                self.players[state.activePlayerNr].observe(self.getObservation(settings, state, state.activePlayerNr, observability = 'fo'))

                #finalize previous move now we know the new state (T(s,a) = s')
                if state.rnd != 0:
                    self.players[state.activePlayerNr].finalize(self.getReward(outcome, settings.playerRoles[state.activePlayerNr]))

                #take new action
                state.actions.append(self.players[state.activePlayerNr].getAction())

                #transit the world state
                if state.actions[-1] != None: #Human agents can take None action  
                    if not simultaniousActions: #transit state per turn
                        self.stateTransition(settings, state, state.actions[-1], state.activePlayerNr)
                        state.actions = []
                    if state.activePlayerNr == settings.nroPlayers-1: #end of round 
                        if simultaniousActions: #transit state per round 
                            for playerNr in xrange(settings.nroPlayers):
                                self.stateTransition(settings, state, state.actions[playerNr], playerNr)
                            state.actions = []
                        state.activePlayerNr = 0
                        state.rnd += 1 
                    else: #next players turn
                        state.activePlayerNr += 1
                        if settings.draw:
                            fpsClock.tick(30)
                    

        #=======================================================================


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
        if settings.playerRoles[playerNr] == PREY and settings.preyTrip == True:
            if random.random() < 0.2: 
                action = STAY
        state.positions[playerNr] = ((state.positions[playerNr][0]+EFFECTS[action][0])%settings.boardSize[0], 
            (state.positions[playerNr][1]+EFFECTS[action][1])%settings.boardSize[1])

    # Check for game end. Game ends if any 2 players share same position.
    def gameEnds(self, settings, state): 
        return len(set(state.positions)) != len(self.players)

    # Determine outcome of a game. Returns 1 if preds win, -1 if prey wins and else 0.
    def getOutcome(self, settings, state): 
        nroPreds = 0
        preds = set()
        players = set()
        for i in xrange(settings.nroPlayers):
            if settings.playerRoles[i] == PREDATOR:
                nroPreds += 1
                preds.add(state.positions[i])
            players.add(state.positions[i])

        if len(preds) != nroPreds: #two predators share position
            return -1
        elif len(players) != len(self.players): #prey is caught
            return 1 
        else:
            return 0

    def getReward(self, outcome, role): 
        reward = outcome * 10
        if role == PREY: #reverse reward
            reward *= -1             
        return reward

    def initState(self, settings, state):
        #give all fixedPos player their position
        positionSet = set()
        positions = [None]*settings.nroPlayers
        for i in xrange(settings.nroPlayers):
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
        state.rnd = 0
        state.activePlayerNr = 0
        state.actions = []

#####################################################################################

class Observation(object):
    """docstring for Observation"""
    def __init__(self, settings, state, playerNr):
        super(Observation, self).__init__()
        self.playerNr = playerNr
        self.roles = settings.playerRoles
        self.positions = state.positions
        self.boardSize = settings.boardSize
        self.rnd = state.rnd


class State(object):
    """docstring for State"""
    def __init__(self):
        super(State, self).__init__()
        self.positions = None
        self.actions = None
        self.episode = 0
        self.rnd = 0 
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
        self.textSurfaceObj = self.fontObj.render('Turn: ' + str(state.rnd), True, BLUE)
        self.displaySurf.blit(self.textSurfaceObj, self.textRectObj)
        self.caughtTextSurf = self.fontObj.render('Caught: ' + str(state.caught), True, BLUE)
        self.displaySurf.blit(self.caughtTextSurf, self.caughtTextRect)
        pygame.display.update()
