import random, sys, pygame, pickle, math, itertools, numpy as np
from pygame.locals import *
from locals import *
import matplotlib.pyplot as plt
# import pylab
# from pylab import *

""" Contains all the different Brain types"""

class Player(object):
    def __init__(self, agent):
        super(Player, self).__init__()
        self.agent = agent 
        self.role = None 
        self.nr = None # player number in game 
        self.observation = None
        self.id = agent.addPlayer(self) # player number in agent 

    def getAction(self):
        return self.agent.getAction(self.id)

    def observe(self, observation):
        self.observation = observation
        self.nr = observation.playerNr
        self.role = observation.roles[self.nr] 
        self.pos = observation.positions[self.nr]
        self.boardSize = observation.boardSize

    def finalize(self, reward):
        self.agent.finalize(reward, self.id) 

    def quit(self):
        self.agent.quit(self.id)

    def finalizeEpisode(self, reward):
        self.agent.finalizeEpisode(reward)

class Agent(object):
    def __init__(self):
        super(Agent, self).__init__()
        self.players = []

    def addPlayer(self, player):
        self.players.append(player)
        return len(self.players)-1

    def getStateRep(self, id):
        state = []
        for i in xrange(len(self.players[id].observation.positions)):
            if i != self.players[id].nr:
                state.append(self.translatePos(self.players[id].pos, self.players[id].observation.positions[i], self.players[id].observation.boardSize))
        return tuple(state) 

    def stateReduce(self, state):
        key = []
        keyTail = []
        for i in xrange(len(state)):
            y,x = state[i]
            if i == 0:
                if y>x:
                    ccRot = -1
                else: 
                    ccRot = 1

            boardSize = (11,11)
            (t_y, t_x) = (boardSize[0]/2, boardSize[1]/2)
            equivalents = []
            equivalents.append((y, x))

            y,x = self.translate(y, x, -t_y, -t_x) 
            equivalents.append(self.translate(x, y, t_y, t_x))   
            for j in xrange(3):
                (y, x) = (-x, y)
                equivalents.append(self.translate(y, x, t_y, t_x))
                equivalents.append(self.translate(x, y, t_y, t_x))

            if i == 0:
                keyHead = sorted(equivalents)[0] 
                key.append(keyHead)
                index = equivalents.index(keyHead)
                rotations = index/2
                mirrored = (index%2) * ccRot
            else:
                keyTail.append(equivalents[index])

        if keyTail != []:
            keyTail = sorted(list(itertools.permutations(keyTail)))[0]
            for pos in keyTail:
                key.append(pos)

        return (tuple(key), rotations, mirrored)

    def translate(self, y, x, t_y, t_x):
        return (y + t_y, x + t_x)

    def transformAction(self, action, rotations, mirrored):
        if mirrored:
            rotations = (rotations + mirrored)%4

        if action == STAY:
            action = STAY
        elif action == UP:
            rotate = [UP, RIGHT, DOWN, LEFT]
            action = rotate[rotations]
        elif action == DOWN:
            rotate = [DOWN, LEFT, UP, RIGHT]
            action = rotate[rotations]
        elif action == LEFT:
            rotate = [LEFT, UP, RIGHT, DOWN]
            action = rotate[rotations]
        elif action == RIGHT:
            rotate = [RIGHT, DOWN, LEFT, UP]
            action = rotate[rotations]
        return action

    def action2Tile(self, action, position, boardSize):
        return ((position[0]+EFFECTS[action][0])%boardSize[0], 
            (position[1]+EFFECTS[action][1])%boardSize[1])

    def tile2Action(self, basePosition, adjacentTile, boardSize):
        if adjacentTile == self.action2Tile(UP, basePosition, boardSize): return UP
        if adjacentTile == self.action2Tile(DOWN, basePosition, boardSize): return DOWN
        if adjacentTile == self.action2Tile(LEFT, basePosition, boardSize): return LEFT
        if adjacentTile == self.action2Tile(RIGHT, basePosition, boardSize): return RIGHT
        if adjacentTile == basePosition: return STAY

    def translatePos(self, ownPos, otherPos, boardSize): 
        #translates other position after centering ownPos 
        center = (boardSize[0]/2, boardSize[1]/2)
        return ((otherPos[0] - ownPos[0] + center[0])%boardSize[0], (otherPos[1] - ownPos[1] + center[1])%boardSize[1])

    def getAction(self): 
        sys.exit("function getAction not implemented for player " + str(player.nr))

    def eGreedy(self, actionValues, epsilon):
        maxval = max(actionValues)
        ind = [i for i, v in enumerate(actionValues) if v == maxval]
        if random.random() >= epsilon:  #select optimal action with 1 - eps prob
            action = ind[int(random.random() * len(ind))] # Non-deterministic in a sense that it choses randomly amongst maxval actions 
        else:   #select random action (includes optimal action)
            action = int(random.random() * len(actionValues))
        return action

    def softMax(self, actionValues, tau):
        if tau == 0: #to prevent devision by zero 
            return self.eGreedy(actionValues, tau) #return greedy action
        else:    
            tempValues = []
            for i in xrange(len(actionValues)):
                tempValues.append(math.e**(actionValues[i]/tau))  
            actionProbs = []
            for i in xrange(len(tempValues)):
                actionProbs.append(tempValues[i]/sum(tempValues))
            #now select an action according to their probability 
            rand = random.random()
            accum = 0
            for i in xrange(len(actionProbs)):
                accum += actionProbs[i]
                if rand < accum:
                    return i

    def finalize(self, R, id):
        pass

    def quit(self, id):
        pass

    def finalizeEpisode(self, reward):
        pass


class OnPolicyMonteCarlo(Agent):
    """Implementation of both On- and Off-Policy Monte-Carlo Control Algorithms"""

    def __init__(self, gamma, epsilon, boardSize):
        super(OnPolicyMonteCarlo,self).__init__()
        self.terminal = (boardSize[0]/2, boardSize[1]/2)
        self.gamma = gamma 
        self.epsilon = epsilon 

        # The collection of states is the cartesian product of the dimensions of the board
        states = [(s,) for s in itertools.product(range(boardSize[0]), range(boardSize[1]))]

        # The collection of state action pairs ins the cartesian product of states and actions
        # TODO generalize over number of agents (now hardcoded for one predator and one prey)
        stateActionPairs = itertools.product(states, range(5))
        self.Q = dict([(s, [0]*5) for s in states if s != self.terminal])            
        self.R = dict([(sa , [0, 0]) for sa in stateActionPairs if sa[0] != self.terminal])
        self.policy = self.randomSoftPolicy(states)
        self.episode = []

    # Return an action, adds state-action pair to episode as side effect.
    def getAction(self, id):
        s = self.getStateRep(id)
        a = self.eGreedy(self.Q[s], self.epsilon)
        self.episode.append((s, a))
        return a

    # Returns a randomly intialized epsilon-soft policy
    def randomSoftPolicy(self, states):
        policy = dict()
        for s in states:
            if s != self.terminal:
                 policy[s] = self._initMoveProbs(s)
        return policy

    # Returns a random epsilon-soft policy for a single state
    def _initMoveProbs(self, s):
        amax = random.randint(0, 4) #Randomly initialise a*
        actions = [0]*5
        actions[amax] = 1
        return self._softMoveProbs(actions)
    
    # Assigns probabilities to actions based on the epsilon soft method.
    # actions must be the list Q(s)
    def _softMoveProbs(self, actions):
        maxi = 0
        amax = 0
        for i in range(len(actions)):
            if actions[i] > maxi:
                maxi = actions[i]
                amax = i
        
        softPol = [None]*len(actions)
        for i in range(len(softPol)): 
            if i == amax:
                softPol[i] = 1 - self.epsilon + self.epsilon / len(softPol)
            else:
                softPol[i] = self.epsilon / len(softPol)
        return softPol

    # Updates the reward 'memory' and Q
    def updateRandQ(self, reward):
        episode = self.episode
        nrOfActions = len(episode)

        # Go throught the episode backwards and update average rewads and Q values.
        indices = range(len(episode) )
        indices.reverse()        
        for i in indices:
            (s, a) = episode[i]
            if (s, a) not in episode[: i - 1]:
                self.R[(s, a)][0] += 1
                self.R[(s, a)][1] = (self.R[(s, a)][1] * (self.R[(s, a)][0] - 1) + math.pow(self.gamma, (nrOfActions - i)) * reward) / self.R[(s,a)][0]
                self.Q[s][a] = self.R[(s, a)][1]
    
    # Update policy with epsilon soft values.      
    def updatePolicy(self):        
        for s in set([s for (s, a) in self.episode]):
            self.policy[s] = self._softMoveProbs(self.Q[s])

            
    # Executes updates at the end of each episode
    def finalizeEpisode(self, reward):
        self.updateRandQ(reward)
        self.updatePolicy()
        #print ''.join(['*']*len(self.episode))
        self.episode = []
        
class TemporalDifference(Agent):
    """docstring for TemporalDifference"""
    def __init__(self, algorithm, QtableFN, Qinitval, alpha, gamma, selParam, selAlgh):
        super(TemporalDifference,self).__init__()
        self.algorithm = algorithm
        self.actionSelection = getattr(self, selAlgh)
        self.QtableFN = QtableFN 
        self.Qinitval = Qinitval
        self.alpha = alpha
        self.gamma = gamma
        self.selParam = selParam # selection parameter Tau or Epsilon
        self.action = None #to store next action
        self.training = 1 

        if not self.training:  # don't select suboptimal actions if not training!
            self.selParam = 0 # amounts to greedy action selection
        try:
            self.Qtable = pickle.load(open( self.QtableFN, "rb"))
            print 'Qtable found and loaded'
        except:
            print 'Qtable not found, creating new'
            self.Qtable = {}
    
    def getAction(self, id): 
        if id == 0: #player 0 plans for all of the agent's players
            self.shape = tuple([5]*len(self.players)) #TODO: preferably init only once!
            self.state = self.getStateRep(id)

            self.state, self.rotations, self.mirrored = self.stateReduce(self.state)#stateReduction
            actionValues = self.Qtable.get(self.state, [self.Qinitval]*5**len(self.players))
            if self.algorithm == 'Q-learning':
                self.action = self.actionSelection(actionValues, self.selParam)
            elif self.algorithm == 'Sarsa':
                if self.action == None: #only true for the first action 
                    self.action = self.actionSelection(actionValues, self.selParam)
            self.actions = np.unravel_index(self.action, self.shape) #tuple with an action for each player
        # return self.actions[id]
        return self.transformAction(self.actions[id], self.rotations, self.mirrored) #stateReduction


    def finalize(self, R, id):
        if id == 0 and self.training == 1:
            self.Qtable[self.state] = self.Qtable.get(self.state, [self.Qinitval]*5**len(self.players))
            sPrime = self.getStateRep(id)
            sPrime, self.rotations, self.mirrored = self.stateReduce(sPrime)#stateReduction
            if self.algorithm == 'Q-learning':
                aPrime = self.actionSelection(self.Qtable.get(sPrime, [self.Qinitval]*5**len(self.players)), 0) #0-epsilon gives max action
            if self.algorithm == 'Sarsa':
                aPrime = self.actionSelection(self.Qtable.get(sPrime, [self.Qinitval]*5**len(self.players)), self.selParam)
            self.Qtable[self.state][self.action] = self.Qtable[self.state][self.action] + self.alpha * (R + self.gamma * self.Qtable.get(sPrime, [self.Qinitval]*5**len(self.players))[aPrime] - self.Qtable[self.state][self.action])
            if self.algorithm == 'Sarsa':
                self.action = aPrime

    def quit(self, id):
        pass
        # if id == 0 and self.training == 1:
        #     try:
        #         pickle.dump(self.Qtable, open('table' + self.algorithm+'a'+str(self.alpha)+'e'+str(self.selParam)+'g'+str(self.gamma)+'i'+str(self.Qinitval)+'.p', "wb" ), pickle.HIGHEST_PROTOCOL)
        #         print 'Qtable successfully saved'
        #     except:
        #         print str(self.players[id].nr)
        #         print "player %s can't write Qtable" % self.players[id].nr

class Human(Agent):
    """docstring for Human"""
    def __init__(self):
        super(Human, self).__init__()
    
    def getAction(self, id):
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

class PseudoRandom(Agent):
    """ Although PsuedoRandom returns random actions for predators it does not for Prey
    Prey never move into a predator and have a given probability to take WAIT action. 
    All other actions have equal probability """
    def __init__(self):
        super(PseudoRandom,self).__init__()

    def getAction(self, id):
        if self.players[id].role == PREY:
            #STAY with a given probability
            rand =random.random()
            if rand < 0.8: 
                return STAY
            else: #find actions that don't cause shared position
                pos = self.players[id].pos
                boardSize = self.players[id].boardSize
                adjacent = set([self.action2Tile(UP, pos, boardSize),
                                self.action2Tile(DOWN, pos, boardSize),
                                self.action2Tile(LEFT, pos, boardSize),
                                self.action2Tile(RIGHT, pos, boardSize)])
                freeAdjacentTiles = adjacent - set(self.players[id].observation.positions)
                possibleActions = []
                for freeAdjacentTile in freeAdjacentTiles:
                    possibleActions.append(self.tile2Action(pos, freeAdjacentTile, boardSize))
                #remaining actions have equal probability
                return possibleActions[int(random.random()*len(possibleActions))] 

        if self.players[id].role == PREDATOR:
            return random.randint(0,4) 

class Random(Agent):
    """docstring for RandomComputer"""
    def __init__(self):
        super(Random,self).__init__()

    def getAction(self, id):
        # return 0
        return random.randint(0,4) 

class DynamicProgramming(Agent):
    """Implementation of Policy Iteration and Value Iteration, works for 2 agent case when fullfilling the role of predator"""

    def __init__(self):
        super(DynamicProgramming,self).__init__()
        self.boardSize = (11,11) #TODO: Solve elegantly!
        Vinitval = 0
        V = np.zeros((self.boardSize[0], self.boardSize[1])) + Vinitval #TODO: only works for 2 agent scenario
        threshold = 0.001 #TODO: change to sensible value
        gamma = 0.8
        terminalState = (self.boardSize[0]/2, self.boardSize[1]/2)
        policy = {}
        deltas = []
        valueIteration = False #stops after one sweep of evaluation and updates V differently (argmax)
        k = None #number of evaluation sweeps per iteration

        
        sweeps = 0 #testing Policy Iteration
        deltas = [] #testing Policy Iteration

        while True:

            #Policy Evaluation
            while k > 0 or k == None:
                sweeps +=1
                delta = 0
                vCopy = V.copy()
                for i in xrange(V.shape[0]): #for each state
                    for j in xrange(V.shape[1]):
                        s = (i, j)
                        v = 0
                        if s != terminalState: #terminal state not part of S
                            policy[s] = policy.get(s, {0: 0.2, 1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2})
                            actionValues = {}
                            for a, p_a in policy[s].items(): #for each action predator considers
                                sPrimes = self.P(s, a)
                                for sPrime, p_sPrime in sPrimes.items():
                                    if valueIteration:
                                        actionValues[a] = actionValues.get(a, 0) + p_sPrime * (self.R(sPrime) + gamma * V[sPrime])
                                    else: 
                                        v += p_a * p_sPrime * (self.R(sPrime) + gamma * V[sPrime])
                            if valueIteration:
                                v = max(actionValues.values()) #key diff between Policy Iteration and Value Iteration!
                            delta = max(delta, abs(v - V[s])) 
                            vCopy[s] = v #V[s] = v
                deltas.append(delta) #testing Policy Iteration
                V = vCopy
                if delta < threshold:
                    break

            #Policy Improvement
            policyChanged = False
            for s in policy:
                v = 0
                if s != terminalState: #terminal state not part of S
                    actionValues = {}
                    for a in xrange(5): #for each possible action 
                        sPrimes = self.P(s,a)
                        for sPrime, p_sPrime in sPrimes.items():
                            actionValues[a] = actionValues.get(a, 0) + p_sPrime * (self.R(sPrime) + gamma * V[sPrime])
                    greedyActions = [i for i, v in actionValues.items() if 
                    v < max(actionValues.values()) + 0.00000001 and v > max(actionValues.values()) - 0.00000001] # small intervall to cope with numerical errors
                    p = 1/float(len(greedyActions))
                    oldPolicy = set(policy[s].keys()) 
                    policy[s] = {}
                    for a in greedyActions:
                        policy[s][a] = p

                    if oldPolicy != set(policy[s].keys()):
                        policyChanged = True

            if not policyChanged or valueIteration:
                break

        self.policy = policy

        #Some printing for the assignments

        # print deltas #testing Policy Iteration
        # print sweeps #testing Policy Iteration

        # fig, sp1 = plt.subplots(figsize=(12,5))
        # sp1.plot(deltas)
        # sp1.set_xlabel('iterations')
        # sp1.set_ylabel('delta')
        # # sp1.set_title('Policy Iteration convergence using gamma of 0.9')
        # pylab.xlim([0,60])
        # pylab.ylim([0,10])
        # plt.show()

        # np.set_printoptions(precision=2)
        # np.set_printoptions(suppress=True)
        # print V
        
        # self.printPolicy(policy)
        # self.latexPrintV(V)
        # sys.exit()


    # Reward function
    def R(self, s): 
        terminalState = (self.boardSize[0]/2, self.boardSize[1]/2)
        if s == terminalState:
            return 10
        else:
            return 0

    # Transition Probabillity function returns a dictionary {s': prob, ...}
    def P(self, s,a): 
        terminalState = (self.boardSize[0]/2, self.boardSize[1]/2)
        if s == terminalState: 
            return {s: 1}
        else:
            halfwayState = ((s[0]+EFFECTS[a][0])%self.boardSize[0], (s[1]+EFFECTS[a][1])%self.boardSize[1])
            if halfwayState == terminalState: #no escape possible
                return {halfwayState: 1}
            else: 
                sPrimes = {}
                for prey_a in xrange(1,5):
                    sPrime =   ((halfwayState[0]+REVEFFECTS[prey_a][0])%self.boardSize[0], 
                                (halfwayState[1]+REVEFFECTS[prey_a][1])%self.boardSize[1])
                    if sPrime != terminalState:
                        sPrimes[sPrime] = None
                prob = 0.2/len(sPrimes) #now we know how many states follow, we know their probs
                for sPrime in sPrimes:
                    sPrimes[sPrime] = prob 
                sPrimes[halfwayState] = 0.8 #prey stays at same location 
                return sPrimes

    def getStateRep(self, id): #@overwrite since we're now interested at state rel to prey
        # self.players[id].pos = observation.positions[self.players[id].nr]
        state = []
        for i in xrange(len(self.players[id].observation.positions)):
            if i != self.players[id].nr:
                state.append(self.translatePos(self.players[id].observation.positions[i], self.players[id].pos, self.boardSize))
        return tuple(state) 

    def getAction(self, id):
        state = self.getStateRep(id)[0] # we know state will be ((y,x),)
        actions = self.policy[state] 
        rand = random.random()
        accum = 0
        for a in actions:
            accum += actions[a] 
            if rand <= accum:
                return a

    def printPolicy(self, p):
        for i in xrange(self.boardSize[0]):
            print 
            for j in xrange(self.boardSize[1]):
                if not (i,j) == (5,5):
                    print p[i,j].keys(),
                else:
                    print 0, 
        print 

    def printV(self, V):
        phi_m = linspace(0, 11, 12)
        phi_p = linspace(0, 11, 12)
        fig, ax = plt.subplots()
        ax.set_xticks(np.arange(0,12,1))
        ax.set_yticks(np.arange(0,12,1))
        plt.grid()

        for i in xrange(11):
            for j in xrange(11):
                v = V[i][j]
                if v == 10: 
                    v = round(v,1) # in order too pretty print
                ax.annotate(str(v),xy=(i+0.25,j+0.35))

        show()

    def latexPrintV(self, V):
        print '\\begin{center}'
        print '\\begin{tabular}{|c|c|c|c|c|c|c|c|c|c|c|}' 
        print '\\hline'
        for i in xrange(11):
            for j in xrange(11):
                if j != 0: 
                    print '&',
                v = V[i][j]
                temp = round(v, 2)
                print  temp, 
            print '\\\ \\hline' 
        print '\\end{tabular}'
        print '\\end{center}'

