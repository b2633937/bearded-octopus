import pygame, sys
from pygame.locals import *
import random
import pickle 
import time #probably obsolete
from locals import *

"""
NOTES: 

-   make sure every player gets reward when game ends! > Qlearning!
-   reset prevstate and prevaction at gameEnds
"""


def main():
    global DISPLAYSURF, AGENTS, BOARDSIZE, TILESIZE, BOARDOFFSETX, BOARDOFFSETY
    pygame.init()
    windowWidth = 800
    windowheight = 600
    BOARDOFFSETX = 250
    BOARDOFFSETY = 50
    BOARDSIZE = 11 # always uneven in order for state to work!
    TILESIZE = 45
    FPS = 30 #frames per second setting
    AGENTS = [] 
    fpsClock = pygame.time.Clock()
    loadSounds()
    loadImages()

    #set up the window
    DISPLAYSURF = pygame.display.set_mode((windowWidth, windowheight), 0, 32)
    gameScreen = GameScreen(DISPLAYSURF)
   
    # instantiate agents
    # AGENTS.append(Human(role='predator', pID=0, bID=0, img=IMAGES['boy']))
    AGENTS.append(RandomComputer(role='prey', pID=1, bID=0, img=IMAGES['princess']))
    #AGENTS.append(RandomComputer(role='prey', pID = 2, bID = 1, img = IMAGES['princess']))
    AGENTS.append(Qlearning(role='predator', pID = 2, bID = 1, img = IMAGES['boy']))
    initPositions(AGENTS, mode = 'random')

    activeAgent = 0
    turn = 0
    caught = 0
    totalTurns = 0
    while True: #the main game loop
        gameScreen.draw(turn, caught)
        handleUserInput() #listen for Quit events etc.
        
        if gameEnds(): #check for game end
            GAMESOUNDS['caught'].play()
            initPositions(AGENTS, mode = 'random')
            caught += 1
            turn = 0
            activeAgent = 0
            #print "===================================="
        else:
            #Agent takes move 
            observation = getObservation(activeAgent, observability = 'fo')
            action = AGENTS[activeAgent].getAction(observation)

            if action != None: 
                #print "------ Active agent: ", activeAgent 
                #print "turn: ", turn
                totalTurns+=1
                print 'total turns: ', totalTurns, 'times caught: ', caught
                transit(activeAgent, action)
                AGENTS[activeAgent].finalize(getReward(AGENTS[activeAgent].role))
                #pass turn to next agent
                if activeAgent == len(AGENTS)-1:
                    activeAgent = 0
                    turn += 1
                else:
                    activeAgent += 1

        if caught % 2 == 0:
            try:
                pickle.dump(AGENTS[1].Qtable, open(AGENTS[1].QtableFN, "wb"), pickle.HIGHEST_PROTOCOL)
            except:
                print "agent %s can't write Qtable" % AGENTS[1].pID
            print AGENTS[1].Qtable
        
        fpsClock.tick(FPS)

def getObservation(agentNr, observability):
    #positions also contains own position!
    if observability == 'fo': #full obervability
        positions = [agent.POS for agent in AGENTS]
    return Observation(positions)

def transit(activeAgent, action):
    AGENTS[activeAgent].POS = action2Tile(action, AGENTS[activeAgent].POS)

def action2Tile(action, position):
    #returns same tile if action is not in {0,1,2,3}
    #STAY is thus returned by this principle rather than explicitly checked!
    X,Y = position
    if action == UP:
        Y = Y-1 if Y > 0 else BOARDSIZE-1
    elif action == DOWN:
        Y = Y+1 if Y < BOARDSIZE-1 else 0
    elif action == LEFT:
        X = X-1 if X > 0 else BOARDSIZE-1
    elif action == RIGHT:
        X = X+1 if X < BOARDSIZE-1 else 0
    return (X,Y)

def tile2Action(basePosition, adjacentTile):
    if adjacentTile == action2Tile(UP, basePosition): return UP
    if adjacentTile == action2Tile(DOWN, basePosition): return DOWN
    if adjacentTile == action2Tile(LEFT, basePosition): return LEFT
    if adjacentTile == action2Tile(RIGHT, basePosition): return RIGHT
    if adjacentTile == basePosition: return STAY

def gameEnds(): #game ends if any 2 agents share same position
    s = set()
    for agent in AGENTS:
        s.add(agent.POS)
    #print 'gameEnds: ', len(s) != len(AGENTS)
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
    IMAGES['boy'] = pygame.image.load('boy.png')
    IMAGES['princess'] = pygame.image.load('princess.png')


#####################################################################################

class Observation(object):
    """docstring for Observation"""
    def __init__(self, positions):
        super(Observation, self).__init__()
        self.positions = positions

class Player(object):
    POS = None
    def __init__(self, role, pID, bID, img):
        super(Player, self).__init__()
        self.role, self.pID, self.bID, self.img = role, pID, bID, img

    def getState(self, observation):
        state = []
        for position in observation.positions:
            if position != self.POS:
                state.append(self.translatePos(self.POS, position))
        return state[0] # TODO: create mp state

    def translatePos(self, ownPos, otherPos):
        #translates other position relative to centered ownPos 
        center = (BOARDSIZE/2, BOARDSIZE/2)
        translation = (center[0] - ownPos[0], center[1] - ownPos[1])
        translOtherPos = [translation[0] + otherPos[0], translation[1] + otherPos[1]]
        if translOtherPos[0] < 0:
            translOtherPos[0] = translOtherPos[0] + BOARDSIZE
        elif translOtherPos[0] >= BOARDSIZE:
            translOtherPos[0] = translOtherPos[0] - BOARDSIZE
        if translOtherPos[1] < 0:
            translOtherPos[1] = translOtherPos[1] + BOARDSIZE
        elif translOtherPos[1] >= BOARDSIZE:
            translOtherPos[1] = translOtherPos[1] - BOARDSIZE
        return tuple(translOtherPos)

    def getAction(self, observation): 
        sys.exit("function getAction not implemented for agent " + str(self.pID))

    def finalize(self, reward):
        pass

    def quit(self):
        pass

class Optimizer(Player):
    def __init__(self, role, pID, bID, img):
        super(Optimizer,self).__init__(role, pID, bID, img) 
        self.criterium = 'maximize' 

    def getAction(self, observation):
        state = self.getState(observation)
        #print state
        x = abs(state[0]-5) 
        y = abs(state[1]-5) 
        #print x, y
        if self.criterium == 'minimize':
            if x == y:
                if random.randint(0,1):
                    if state[0]-5 > 0:
                        return RIGHT
                    else:
                        return LEFT
                else:
                    if state[1]-5 > 0:
                        return DOWN
                    else:
                        return UP
            elif x > y:
                if state[0]-5 > 0:
                    return RIGHT
                else:
                    return LEFT
            else:
                if state[1]-5 > 0:
                    return DOWN
                else:
                    return UP
        if self.criterium == 'maximize':
            if x == y:
                if random.randint(0,1):
                    if state[0]-5 > 0:
                        return LEFT
                    else:
                        return RIGHT
                else:
                    if state[1]-5 > 0:
                        return UP
                    else:
                        return DOWN
            elif x > y:
                if state[0]-5 > 0:
                    return LEFT
                else:
                    return RIGHT
            else:
                if state[1]-5 > 0:
                    return UP
                else:
                    return DOWN
        return None


class Human(Player):
    def __init__(self, role, pID, bID, img):
        super(Human,self).__init__(role, pID, bID, img)
    
    def getAction(self, observation):
        for event in pygame.event.get():
            if event.type == KEYDOWN:
                if event.key == K_UP:
                    return UP
                elif event.key == K_DOWN:
                    return DOWN
                elif event.key == K_LEFT:
                    return LEFT
                elif event.key == K_RIGHT:
                    return RIGHT
                elif event.key == K_RETURN:
                    return STAY
        return None

class RandomComputer(Player):
    def __init__(self, role, pID, bID, img):
        super(RandomComputer,self).__init__(role, pID, bID, img)

    def getAction(self, observation):
        #STAY with a given probability
        rand = random.random()
        if rand <= 0.8: #TODO: change to 0.8!
            return STAY
        else: #find actions that don't cause shared position
            adjacent = set([action2Tile(UP, self.POS),
                            action2Tile(DOWN, self.POS),
                            action2Tile(LEFT, self.POS),
                            action2Tile(RIGHT, self.POS)])
            freeAdjacentTiles = adjacent - set(observation.positions)
            possibleActions = []
            for freeAdjacentTile in freeAdjacentTiles:
                possibleActions.append(tile2Action(self.POS, freeAdjacentTile))
            #remaining actions have equal probability
            rand = random.random()
            chance = float(1) / len(possibleActions)
            return possibleActions[int(rand / chance)] 

class Qlearning(Player):
    """docstring for Qlearning"""
    def __init__(self, role, pID, bID, img):
        super(Qlearning,self).__init__(role, pID, bID, img)
        self.QtableFN = 'Qtable.p'
        self.Qinitval = [0, 0, 0, 0, 0] 
        self.epsilon = 0.05 # TODO: change into 0.05'ish
        self.alpha = 0.3
        self.gamma = 0.9
        self.prevState, self.prevAction, self.lastState, self.lastAction = None, None, None, None 
        try:
            self.Qtable = pickle.load(open( QtableFN, "rb"), pickle.HIGHEST_PROTOCOL)
        except:
            #print 'Qtable not found, creating new'
            self.Qtable = {}
    
    def getAction(self, observation): #Qinitval name in AA?
        self.state = self.getState(observation)
        actions = self.Qtable.get(self.state, list(self.Qinitval))
        maxval = max(actions)

        ind = [i for i, v in enumerate(actions) if v != maxval]
        if random.random() < self.epsilon and len(ind) != 0:
            #select sub optimal action with eps prob if present
            rand = random.random()
            chance = float(1) / len(ind)
            self.action = ind[int(rand / chance)]
            #print 'returning from suboptimal actions', ind, 'action', self.action
        else:
            #select from max actions with 1-eps prob
            ind = [i for i, v in enumerate(actions) if v == maxval]
            rand = random.random()
            chance = float(1) / len(ind)
            self.action = ind[int(rand / chance)]
            #print 'returning from max actions', ind, 'action', self.action
        return self.action

    def finalize(self, reward):
        #Qlearn
        self.Qtable[self.state] = self.Qtable.get(self.state, list(self.Qinitval))
        if self.prevAction != None:
            self.Qtable[self.prevState][self.prevAction] = self.Qtable[self.prevState][self.prevAction] + self.alpha * (reward + self.gamma * self.Qtable[self.state][self.action] - self.Qtable[self.prevState][self.prevAction])
        self.prevState = self.state
        self.prevAction = self.action
        # #print self.Qtable

    def quit():
        try:
            pickle.dump(self.Qtable, open( self.QtableFN, "wb" ), pickle.HIGHEST_PROTOCOL)
        except:
            print "agent %s can't write Qtable" % self.pID



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


