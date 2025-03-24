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

        "*** student work ***"
        self.q_values = util.Counter()


    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** student work ***"
        return self.q_values[(state, action)]


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** student work ***"
        legalActions = self.getLegalActions(state)
        if len(legalActions)==0:
          return 0.0
        tmp = util.Counter()
        for action in legalActions:
          tmp[action] = self.getQValue(state, action)
        return tmp[tmp.argMax()] 

          




    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** student work ***"
        actions = self.getLegalActions(state)
        bestA = None
        maxVal = float('-inf')
        for action in actions:
          q_value = self.q_values[(state, action)]
          if maxVal < q_value:
            maxVal = q_value
            bestA = action
        return bestA


    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob = 0.1) to get a True value prob percentage of the times.
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        "*** student work ***"
        action = util.flipCoin(self.epsilon)
        
        if action:
          return random.choice(legalActions)
        else:
            return self.getPolicy(state)
        # return action



    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** student work ***"
        oldqvalue = self.getQValue(state, action)
        oldpart = (1 - self.alpha) * oldqvalue
        reward_part = self.alpha * reward
        if not nextState:
          self.q_values[(state, action)] = oldpart + reward_part
        else:
          nextState_part = self.alpha * self.discount * self.getValue(nextState)
          self.q_values[(state, action)] = oldpart + reward_part + nextState_part
        """
          QLearning update algorithm:
          Q(s,a) = (1-alpha)*Q(s,a) + alpha*sample
          ***sample = R(s,a,s') + gamma*max(Q(s',a'))***
        """


    def getPolicy(self, state):
        "Returns the policy at the state."
        "*** student work ***"
        return self.computeActionFromQValues(state)


    def getValue(self, state):
        return self.computeValueFromQValues(state)

