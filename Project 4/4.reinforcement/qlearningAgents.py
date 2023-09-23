# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        self.qValues = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        return self.qValues[(state, action)]

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        actions = self.getLegalActions(state)
        if not actions:
            return 0
        maxVal = -float('inf')
        for action in actions:
            maxVal = max(maxVal, self.getQValue(state, action))
        return maxVal

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        maxVal, maxActions = -float('inf'), []
        actions = self.getLegalActions(state)

        for action in actions:
            qValue = self.getQValue(state, action)
            if qValue > maxVal:
                maxVal, maxActions = qValue, [action]
            elif qValue == maxVal:
                maxActions += [action]
        return random.choice(maxActions) if maxActions else None

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
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        if not legalActions:
            return None
        return random.choice(legalActions) if util.flipCoin(self.epsilon) else self.computeActionFromQValues(state)

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        weightedOldVal = (1 - self.alpha) * self.getQValue(state, action)
        nextAction = self.computeActionFromQValues(nextState)
        weightedSample = self.alpha * (reward + (self.discount * self.getQValue(nextState, nextAction)))
        self.qValues[(state, action)] = weightedOldVal + weightedSample

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


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
        self.index = 0  # This is always Pacman
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
    def __init__(self, extractor='IdentityExtractor', runCount=1, **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()
        self.episodeRewardMap = util.Counter()
        self.runCount = runCount

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        return self.weights * self.featExtractor.getFeatures(state, action)

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        currQVal = self.getQValue(state, action)
        sampleVal = reward + (self.discount * self.computeValueFromQValues(nextState))
        features = self.featExtractor.getFeatures(state, action)
        for feature, value in features.items():
            self.weights[feature] += (self.alpha * (sampleVal - currQVal) * value)

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        self.episodeRewardMap[self.episodesSoFar + 1] = state.getScore()
        PacmanQAgent.final(self, state)
        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            import pickle
            with open(f'Runs/{self.runCount}.pickle', 'wb') as f:
                pickle.dump(self.episodeRewardMap, f, protocol=pickle.HIGHEST_PROTOCOL)
            pass


# class SarsaAgent(QLearningAgent):
#     def __init__(self, **args):
#         super().__init__(**args)
#         self.nextAction = None
#
#     def getAction(self, state):
#         if self.nextAction is None:
#             self.nextAction = QLearningAgent.getAction(self, state)
#         return self.nextAction
#
#     def update(self, state, action, nextState, reward):
#         weightedOldVal = (1 - self.alpha) * self.getQValue(state, action)
#         nextAction = QLearningAgent.getAction(self, nextState)
#         if not nextAction:
#             return
#         weightedSample = self.alpha * (reward + (self.discount * self.getQValue(nextState, nextAction)))
#         self.qValues[(state, action)] = weightedOldVal + weightedSample
#         self.nextAction = nextAction
#
#
# class PacmanSarsaAgent(SarsaAgent):
#     def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.2, numTraining=0, **args):
#         args['epsilon'] = epsilon
#         args['gamma'] = gamma
#         args['alpha'] = alpha
#         args['numTraining'] = numTraining
#         self.index = 0  # This is always Pacman
#         SarsaAgent.__init__(self, **args)
#
#     def getAction(self, state):
#         """
#         Simply calls the getAction method of QLearningAgent and then
#         informs parent of action for Pacman.  Do not change or remove this
#         method.
#         """
#         action = SarsaAgent.getAction(self, state)
#         self.doAction(state, action)
#         return action

class SarsaAgent(QLearningAgent):
    def __init__(self, **args):
        super().__init__(**args)
        self.currentAction = None

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
        action = self.getCurrentAction(state)
        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here
          NOTE: You should never call this function,
          it will be called on your behalf
        """
        nextAction = self.epsilonGreedyAction(nextState)
        oldVal = (1 - self.alpha) * self.getQValue(state, action)
        sampleVal = self.alpha * (reward + self.discount * self.getQValue(nextState, nextAction))
        self.qValues[(state, action)] = oldVal + sampleVal
        self.setCurrentAction(nextAction)

    def getCurrentAction(self, state=None):
        if self.currentAction is None:
            self.currentAction = self.epsilonGreedyAction(state)
        return self.currentAction

    def setCurrentAction(self, action):
        self.currentAction = action

    def epsilonGreedyAction(self, state):
        legalActions = self.getLegalActions(state)
        if not legalActions:
            return None

        if util.flipCoin(self.epsilon):
            return random.choice(legalActions)
        else:
            return self.computeActionFromQValues(state)


class PacmanSarsaAgent(SarsaAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.2, numTraining=0, **args):
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
        self.index = 0  # This is always Pacman
        SarsaAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """

        action = SarsaAgent.getAction(self, state)
        self.doAction(state, action)
        return action

class EpisodicSemiGradient(PacmanSarsaAgent):
    """
       ApproximateQLearningAgent
       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """

    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanSarsaAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        if not action:
            return 0
        return self.weights * self.featExtractor.getFeatures(state, action)

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        nextAction = self.epsilonGreedyAction(nextState)
        currQVal = self.getQValue(state, action)
        sampleVal = reward + (self.discount * self.getQValue(nextState, nextAction))
        features = self.featExtractor.getFeatures(state, action)
        for feature, value in features.items():
            self.weights[feature] += (self.alpha * (sampleVal - currQVal) * value)
        self.setCurrentAction(nextAction)

    def final(self, state):
        PacmanSarsaAgent.final(self, state)
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            # print(self.weights)
            pass
