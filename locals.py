import matplotlib.pyplot as plt
import numpy as np
import pygame

#------------------- define COLORS -------------------
WHITE    = (255, 255, 255)
DARKGRAY = ( 70,  70,  70)
BLACK    = (  0,   0,   0)
RED      = (255,   0,   0)
GREEN    = (  0, 255,   0)
BLUE     = (  0,   0, 255)
YELLOW   = (255, 255,   0)
ORANGE   = (255, 128,   0)
PURPLE   = (255,   0, 255)

#-------- available ACTIONS and their EFFECTS ---------
UP = 1
DOWN = 2 
LEFT = 3
RIGHT = 4 
STAY = 0
# EFFECTS = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
EFFECTS = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
REVEFFECTS = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)] #when making moves in an agents StateRep

#--------------- load IMAGES and SOUNDS ---------------
IMAGES = {}
#load and scale background image
backgroundImg = pygame.image.load('images/flippyboard.png')
IMAGES['backgroundImg'] =  backgroundImg 
#load agent images
IMAGES['boy'] = pygame.image.load('images/princess.png')
IMAGES['princess'] = pygame.image.load('images/boy.png')

SOUNDS = {}
pygame.mixer.init()
SOUNDS['caught'] = pygame.mixer.Sound('sounds/match4.wav')


#----- function used for plotting error margins -------
def errorfill(x, y, yerr, color=None, alpha_fill=0.3, ax=None):
    ax = ax if ax is not None else plt.gca()
    if color is None:
        color = ax._get_lines.color_cycle.next()
    if np.isscalar(yerr) or len(yerr) == len(y):
        ymin = y - yerr
        ymax = y + yerr
    elif len(yerr) == 2:
        ymin, ymax = yerr
    ax.plot(x, y, color=color)
    ax.fill_between(x, ymax, ymin, color=color, alpha=alpha_fill)