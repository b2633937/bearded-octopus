import pygame
from locals import *

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
    if mode == 'fixed':
        AGENTS[0].POS = (5,5)
        AGENTS[1].POS = (0,0) 

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
    GAMESOUNDS = {}
    GAMESOUNDS['caught'] = pygame.mixer.Sound('match4.wav')
    return GAMESOUNDS

def loadImages():
    IMAGES = {}
    #load and scale background image
    backgroundImg = pygame.image.load('flippyboard.png')
    backgroundImg = pygame.transform.scale(backgroundImg, (495,495))
    IMAGES['backgroundImg'] =  backgroundImg 
    #load agent images
    IMAGES['boy'] = pygame.image.load('boy.png')
    IMAGES['princess'] = pygame.image.load('princess.png')
    return IMAGES
