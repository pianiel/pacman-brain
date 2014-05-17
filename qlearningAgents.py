from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math
         
class QLearningAgent(ReinforcementAgent):
  """
    Q-Learning Agent
    
    Functions you should fill in:
      - getQValue
      - getAction
      - getValue
      - getPolicy
      - update
      
    Instance variables you have access to
      - self.epsilon (exploration prob)
      - self.alpha (learning rate)
      - self.gamma (discount rate)
    
    Functions you should use
      - self.getLegalActions(state) 
        which returns legal actions
        for a state
  """
  def __init__(self, **args):
    "You can initialize Q-values here..."
    ReinforcementAgent.__init__(self, **args)

    "*** YOUR CODE HERE ***"
    self.Q = {}
    # self.ext = SimpleExtractor()

  def getThingDirection(self, p1, p2):
    x, y = p2[0]-p1[0], p2[1]-p1[1]
    if y > x:
        if y > -x:
            return Directions.NORTH
        else:
            return Directions.WEST
    else:
        if y > -x:
            return Directions.EAST
        else:
            return Directions.SOUTH

    return

  def closestGhost(self, state):
    pos = state.getPacmanPosition()
    gpositions = state.getGhostPositions()
    gdists = [util.manhattanDistance(pos, gpos) for gpos in gpositions]
    idx, closestGhostDist = min(enumerate(gdists), key=lambda p: p[1])
    if closestGhostDist > 4.5:
        closestGhostDist = 'Far'
    # closestGhostDir = 0 # f(pos, gpositions[idx])
    return gpositions[idx], closestGhostDist

  def getSimplifiedState(self, state):
    #TODO
    remainingFood = state.getNumFood()
    pos = state.getPacmanPosition()
    cfood = closestFood(pos, state.getFood(), state.getWalls(), withPos=True)
    if cfood is None:
        closestFoodDist = None
        closestFoodDir = None
    else:
        closestFoodDist = cfood[1]
        closestFoodDir = self.getThingDirection(pos, cfood[0])
        if closestFoodDist > 4.5:
            closestFoodDir = 'Far'
    cghost = self.closestGhost(state)
    if cghost is None:
        closestGhostDist = None
        closestGhostDir = None
    else:
        closestGhostDist = cghost[1]
        closestGhostDir = self.getThingDirection(pos, cghost[0])
    result = (remainingFood, closestFoodDist, 'foodOn:', closestFoodDir, closestGhostDist, 'ghostOn:', closestGhostDir, 'goOn:')
    return result
    # return self.ext.getFeatures(state, action)

  def getQValue(self, state, action):
    """
      Returns Q(state,action)    
      Should return 0.0 if we never seen
      a state or (state,action) tuple 
    """
    "*** YOUR CODE HERE ***"
    simplified_state = self.getSimplifiedState(state)
    if (simplified_state, action) not in self.Q:
        return 0.0
    return self.Q[(simplified_state, action)]
    
  def getValue(self, state):
    """
      Returns max_action Q(state,action)        
      where the max is over legal actions.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return a value of 0.0.
    """
    "*** YOUR CODE HERE ***"
    legalActions = self.getLegalActions(state);
    max_action = float ('-inf')
    for action in legalActions:
        max_action = max (max_action, self.getQValue(state, action))
    if max_action == float ('-inf'):
        return 0.0
    return max_action
    
  def getPolicy(self, state):

    """
      Compute the best action to take in a state.  Note that if there
      are no legal actions, which is the case at the terminal state,
      you should return None.
    """
    "*** YOUR CODE HERE ***"
    max_action = float ('-inf')
    best_action = None
    for action in self.getLegalActions(state):
        candidate = self.getQValue (state, action)
        if max_action < candidate:
            max_action = candidate
            best_action = action
    return best_action
    
  def getAction(self, state):
    """
      Compute the action to take in the current state.  With
      probability self.epsilon, we should take a random action and
      take the best policy action otherwise.  Note that if there are
      no legal actions, which is the case at the terminal state, you
      should choose None as the action.
    
      HINT: You might want to use util.flipCoin(prob)
      HINT: To pick randomly from a list, use random.choice(list)
    """  
    "*** YOUR CODE HERE ***"
    # Pick Action
    legalActions = self.getLegalActions(state)
    if not legalActions:
        return None
    if util.flipCoin(self.epsilon):
        return random.choice(legalActions)
    return self.getPolicy(state)
  
  def update(self, state, action, nextState, reward):
    """
      The parent class calls this to observe a 
      state = action => nextState and reward transition.
      You should do your Q-Value update here
      
      NOTE: You should never call this function,
      it will be called on your behalf
    """
    "*** YOUR CODE HERE ***"
    self.Q[(self.getSimplifiedState(state), action)] = self.getQValue(state, action) + self.alpha * (reward + self.gamma * self.getValue(nextState) - self.getQValue(state, action))
    
class PacmanQAgent(QLearningAgent):
  "Exactly the same as QLearningAgent, but with different default parameters"
  
  def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
    """
    These default parameters can be changed from the pacman.py command line.
    For example, to change the exploration rate, try:
        python pacman.py -p PacmanQLearningAgent -a epsilon=0.1
    
    alpha    - learning rate
    epsilon  - exploration rate
    gamma    - discount factor
    numTraining - number of training episodes, i.e. no learning after these many episodes
    """
    args['epsilon'] = epsilon
    args['gamma'] = gamma
    args['alpha'] = alpha
    args['numTraining'] = numTraining
    QLearningAgent.__init__(self, **args)

  def getAction(self, state):
    """
    Simply calls the getAction method of QLearningAgent and then
    informs parent of action for Pacman.  Do not change or remove this
    method.
    """
    action = QLearningAgent.getAction(self,state)
    self.doAction(state,action)
    return action

    
class ApproximateQAgent(PacmanQAgent):
  """
     ApproximateQLearningAgent
     
     You should only have to overwrite getQValue
     and update.  All other QLearningAgent functions
     should work as is.
  """
  def __init__(self, extractor='IdentityExtractor', **args):
    self.featExtractor = util.lookup(extractor, globals())()
    PacmanQAgent.__init__(self, **args)

    # You might want to initialize weights here.
    "*** YOUR CODE HERE ***"
    
  def getQValue(self, state, action):
    """
      Should return Q(state,action) = w * featureVector
      where * is the dotProduct operator
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()
    
  def update(self, state, action, nextState, reward):
    """
       Should update your weights based on transition  
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()
    
  def final(self, state):
    "Called at the end of each game."
    # call the super-class final method
    PacmanQAgent.final(self, state)
    
    # did we finish training?
    if self.episodesSoFar == self.numTraining:
      # you might want to print your weights here for debugging
      "*** YOUR CODE HERE ***"
      pass
