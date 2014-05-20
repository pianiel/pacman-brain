"Feature extractors for Pacman game states"

from game import Directions, Actions
from collections import deque
import util

class FeatureExtractor:  
  def getFeatures(self, state, action):    
    """
      Returns a dict from features to counts
      Usually, the count will just be 1.0 for
      indicator functions.  
    """
    util.raiseNotDefined()

class IdentityExtractor(FeatureExtractor):
  def getFeatures(self, state, action):
    feats = util.Counter()
    feats[(state,action)] = 1.0
    return feats


def closestFood(pos, food, walls, withPos=False):
  """
  closestFood -- this is similar to the function that we have
  worked on in the search project; here its all in one place
  """
  fringe = [(pos[0], pos[1], 0)]
  expanded = set()
  while fringe:
    pos_x, pos_y, dist = fringe.pop(0)
    if (pos_x, pos_y) in expanded:
      continue
    expanded.add((pos_x, pos_y))
    # if we find a food at this location then exit
    if food[pos_x][pos_y]:
      if withPos:
        return (pos_x, pos_y), dist
      return dist
    # otherwise spread out from the location to its neighbours
    nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
    for nbr_x, nbr_y in nbrs:
      fringe.append((nbr_x, nbr_y, dist+1))
  # no food found
  return None

def closestFeatures(state):
    (px, py) = state.getPacmanPosition()
    ghost_positions = state.getGhostPositions()
    food = state.getFood()
    capsules_positions = state.getCapsules()
    walls = state.getWalls()

    food_dist = None
    food_dir = None
    ghost_dist = None
    ghost_dir = None
    capsule_dist = None
    capsule_dir = None

    look_for_food = state.getNumFood > 0
    look_for_ghost = state.getNumAgents > 1
    look_for_capsule = len(state.getCapsules()) > 0

    to_visit = deque()
    to_visit.append((px, py, 0, None))
    visited = set()

    while (look_for_food or look_for_ghost or look_for_capsule) and to_visit:
        # dirr, because dir is a builtin function...
        x, y, dist, dirr = to_visit.popleft()
        visited.add((x, y))

        if look_for_food:
            if food[x][y]:
                food_dist = dist
                food_dir = dirr
                look_for_food = False

        if look_for_ghost:
            if (x, y) in ghost_positions:
                ghost_dist = dist
                ghost_dir = dirr
                look_for_ghost = False

        if look_for_capsule:
            if (x, y) in capsules_positions:
                capsule_dist = dist
                capsule_dir = dirr
                look_for_capsule = False

        nbrs = Actions.getLegalNeighbors((x, y), walls)
        for (nbr_x, nbr_y) in nbrs:
            if (nbr_x, nbr_y) in visited:
                continue

            if dirr is None:
                nbr_dir = getThingDirection((x, y), (nbr_x, nbr_y))

            next_dir = dirr is not None and dirr or nbr_dir
            to_visit.append((nbr_x, nbr_y, dist + 1, next_dir))

    return {'food': (food_dist, food_dir),
            'ghost': (ghost_dist, ghost_dir),
            'capsule': (capsule_dist, capsule_dir)}

def getThingDirection(p1, p2):
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

class SimpleExtractor(FeatureExtractor):
  """
  Returns simple features for a basic reflex Pacman:
  - whether food will be eaten
  - how far away the next food is
  - whether a ghost collision is imminent
  - whether a ghost is one step away
  """
  
  def getFeatures(self, state, action):
    # extract the grid of food and wall locations and get the ghost locations
    food = state.getFood()
    walls = state.getWalls()
    ghosts = state.getGhostPositions()

    features = util.Counter()
    
    features["bias"] = 1.0
    
    # compute the location of pacman after he takes the action
    x, y = state.getPacmanPosition()
    dx, dy = Actions.directionToVector(action)
    next_x, next_y = int(x + dx), int(y + dy)
    
    # count the number of ghosts 1-step away
    features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)

    # if there is no danger of ghosts then add the food feature
    if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
      features["eats-food"] = 1.0
    
    dist = closestFood((next_x, next_y), food, walls)
    if dist is not None:
      # make the distance a number less than one otherwise the update
      # will diverge wildly
      features["closest-food"] = float(dist) / (walls.width * walls.height) 
    features.divideAll(10.0)
    return features