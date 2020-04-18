# featureExtractors.py
# --------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"Feature extractors for Pacman game states"

from game import Directions, Actions
import util
from pacman import PacmanRules

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

class CoordinateExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[state] = 1.0
        feats['x=%d' % state[0]] = 1.0
        feats['y=%d' % state[0]] = 1.0
        feats['action=%s' % action] = 1.0
        return feats

def closestFood(pos, food, walls):
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
            return dist
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist+1))
    # no food found
    return None

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

class NewExtractor(FeatureExtractor):
    """
    Design you own feature extractor here. You may define other helper functions you find necessary.
    """
    def getFeatures(self, state, action):
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostPositions()
        features = util.Counter()

        features["bias"] = 1.0

        # compute the location of pacman after action
        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        # feature for number of ghosts 1-step away
        features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)

        # feature for number of ghosts 2-steps away
        features["#-of-ghosts-2-steps-away"] = 0.0
        copied_state = state.deepCopy()
        PacmanRules.applyAction(copied_state, action)
        next_legal_actions = copied_state.getLegalPacmanActions()

        for next_action in next_legal_actions:
            next_dx, next_dy = Actions.directionToVector(next_action)
            next_next_x, next_next_y = int(next_x + next_dx), int(next_y + next_dy)
            features["#-of-ghosts-2-steps-away"] += sum((next_next_x, next_next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)
            
        if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
            features["eats-food"] = 1.0

        # closest food feature
        dist = closestFood((next_x, next_y), food, walls)
        if dist is not None:
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["closest-food"] = float(dist) / (walls.width * walls.height)

        def manhattanDistance( pos1, pos2 ):
            return abs( pos1[0] - pos2[0] ) + abs( pos1[1] - pos2[1] )

        
        # closest scared ghosts feature
        scared_ghosts = list(filter(lambda x: x.scaredTimer != 0, state.getGhostStates()))
        if scared_ghosts:
            scared_ghosts_manhattan_distances = list(map(lambda x: manhattanDistance(state.getPacmanPosition(), x.getPosition()), scared_ghosts))
            closest_distance_scared_ghosts = min(scared_ghosts_manhattan_distances)
            features["closest_scared_ghosts"] = closest_distance_scared_ghosts

        # inverse closest active ghosts feature
        active_ghosts = list(filter(lambda x: x.scaredTimer == 0, state.getGhostStates()))
        if active_ghosts:
            active_ghosts_manhattan_distances = list(map(lambda x: manhattanDistance(state.getPacmanPosition(), x.getPosition()), active_ghosts))
            closest_distance_active_ghosts = min(active_ghosts_manhattan_distances)
            if closest_distance_active_ghosts != 0:
                features["inverse_closest_active_ghosts"] = 1 / closest_distance_active_ghosts

        features.divideAll(10.0)
        return features
