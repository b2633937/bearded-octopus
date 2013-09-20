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
STAY = 0
UP = 1
DOWN = 2 
LEFT = 3
RIGHT = 4 
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


