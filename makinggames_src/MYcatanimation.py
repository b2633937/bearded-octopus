import pygame, sys
from pygame.locals import *
import random
import time # probably obsolete
from locals import *


def main():
    global DISPLAYSURF, AGENTS, BOARDSIZE, TILESIZE, BOARDOFFSETX, BOARDOFFSETY
    pygame.init()
    windowWidth = 800
    windowheight = 600
    BOARDOFFSETX = 250
    BOARDOFFSETY = 50
    BOARDSIZE = 10
    TILESIZE = 50
    PlayersTurn = True
    FPS = 30 # frames per second setting
    AGENTS = [] 
    fpsClock = pygame.time.Clock()
    loadSounds()
    loadImages()

    # set up the window
    DISPLAYSURF = pygame.display.set_mode((windowWidth, windowheight), 0, 32)
    gameScreen = GameScreen(DISPLAYSURF)
   
    # instantiate agents
    AGENTS.append(Human(pID = 0, bID = 0, img = IMAGES['boy']))
    AGENTS.append(RandomComputer(pID = 1, bID = 1, img = IMAGES['princess']))
    initPositions(AGENTS)

    activeAgent = 0
    turn = 0
    caught = 0
    while True: # the main game loop

        gameScreen.draw(0,0)
        handleUserInput() # listen for Quit events etc.
       
        #Agent takes move 
        action = AGENTS[activeAgent].getAction('observation')
        if action != None:
            X,Y = AGENTS[activeAgent].POS
            if action == UP:
                Y = Y-1 if Y > 0 else BOARDSIZE-1
            if action == DOWN:
                Y = Y+1 if Y < BOARDSIZE-1 else 0
            if action == LEFT:
                X = X-1 if X > 0 else BOARDSIZE-1
            if action == RIGHT:
                X = X+1 if X < BOARDSIZE-1 else 0
            AGENTS[activeAgent].POS = (X,Y)

            #pass turn to next agent
            if activeAgent == len(AGENTS)-1:
                activeAgent = 0
                turn += 1
            else:
                activeAgent += 1

        # check for game end
        if AGENTS[0].POS == AGENTS[1].POS:
            GAMESOUNDS['caught'].play()
            initPositions(AGENTS)
            caught += 1
            turn = 0
            activeAgent = 0

        fpsClock.tick(FPS)

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
    screenY += 9 # TODO: handle stupid picture size ofsett
    DISPLAYSURF.blit(img, img.get_rect(center=(screenX,screenY)).topleft)

def drawBoard():
    pygame.draw.rect(DISPLAYSURF, RED, (250, 50, 500, 500), 3)
    for i in xrange(1, BOARDSIZE):
        pygame.draw.line(DISPLAYSURF, RED, (250, 50+i*(500/BOARDSIZE)), (250+500, 50+i*(500/BOARDSIZE)),1)
        pygame.draw.line(DISPLAYSURF, RED, (250+i*(500/BOARDSIZE), 50), (250+i*(500/BOARDSIZE), 50+500),1)
    DISPLAYSURF.blit(DISPLAYSURF, (0, 0))

def initPositions(AGENTS): #TODO: prevent same POS
    for agent in AGENTS:
        agent.POS = (random.randint(0,9), random.randint(0,9))

def loadSounds():
    global GAMESOUNDS
    GAMESOUNDS = {}
    GAMESOUNDS['caught'] = pygame.mixer.Sound('match4.wav')

def loadImages():
    global IMAGES
    IMAGES = {}
    #load and scale background image
    backgroundImg = pygame.image.load('flippyboard.png')
    backgroundImg = pygame.transform.scale(backgroundImg, (500,500))
    IMAGES['backgroundImg'] =  backgroundImg 
    #load agent images
    IMAGES['boy'] = pygame.image.load('boy.png')
    IMAGES['princess'] = pygame.image.load('princess.png')


#####################################################################################

class Player(object):
    POS = None
    def __init__(self, pID, bID, img):
        super(Player, self).__init__()
        self.pID, self.bID, self.img = pID, bID, img
        

class Human(Player):
    def __init__(self, pID, bID, img):
        super(Human,self).__init__(pID, bID, img)
    
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
        return None

class RandomComputer(Player):
    def __init__(self, pID, bID, img):
        super(RandomComputer,self).__init__(pID, bID, img)
    
    def getAction(self, observation):
        time.sleep(0.2)
        action = random.randint(0,3)
        return action

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
        self.quitTextRect.topleft = (20, 500)

    def draw(self, turn, caught):
        DISPLAYSURF.fill(WHITE)
        DISPLAYSURF.blit(IMAGES['backgroundImg'], (250, 50))
        drawBoard() 
        for agent in AGENTS:
            draw2Tile(agent.POS, agent.img)
        DISPLAYSURF.blit(self.quitTextSurf, self.quitTextRect)
        self.textSurfaceObj = self.fontObj.render('Turn: ' + str(turn), True, BLUE)
        DISPLAYSURF.blit(self.textSurfaceObj, self.textRectObj)
        self.caughtTextSurf = self.fontObj.render('Caught: ' + str(caught), True, BLUE)
        DISPLAYSURF.blit(self.caughtTextSurf, self.caughtTextRect)
        pygame.display.update()


class observation(object):
    """docstring for observation"""
    def __init__(self, arg):
        super(observation, self).__init__()
        self.positions = arg
        
#####################################################################################

if __name__ == '__main__':
    main()


