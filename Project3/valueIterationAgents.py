# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, new_state)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for i in range(self.iterations):
            self.iterations -= 1
            state_lst = util.Counter() # store new value of a state
            new_state_lst = util.Counter() # store whether a state has been updated
            for state in self.mdp.getStates():
                action_lowest = self.computeActionFromValues(state)
                if action_lowest:
                    new_value = self.computeQValueFromValues(state, action_lowest)
                    state_lst[state] = new_value
                    new_state_lst[state] = 1
            for state in self.mdp.getStates():
                if new_state_lst[state]:
                    self.values[state] = state_lst[state]


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        answer = 0

        state_prob = self.mdp.getTransitionStatesAndProbs(state, action)
        for new_state, prob in state_prob:
            reward = self.mdp.getReward(state, action, new_state)
            answer += prob * (reward + (self.discount * self.getValue(new_state)))
        return answer
        # page 6 Q-learnnig. compute q* with given v*

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values old_valently stored in self.values.
          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        bestAct = None
        reward_best = - float ('inf')
        for action in self.mdp.getPossibleActions(state):
            qvalue = self.computeQValueFromValues(state, action)
            if qvalue > reward_best:
                reward_best = qvalue
                bestAct = action
        return bestAct


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the total_states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)


    def runValueIteration(self):
        total_states_all = self.mdp.getStates()
        length = len(total_states_all)
        count = 0
        for i in range(self.iterations):
            self.iterations -= 1
            state_lst = total_states_all[count%length]
            count += 1
            action_lowest = self.computeActionFromValues(state_lst)
            if action_lowest:
                self.values[state_lst] = self.computeQValueFromValues(state_lst, action_lowest)

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*
        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        lst_pre = {}
        total_states = self.mdp.getStates()

        for state in total_states:
            if not self.mdp.isTerminal(state):
                for action in self.mdp.getPossibleActions(state):
                    for new_state, probs in self.mdp.getTransitionStatesAndProbs(state, action):
                        if new_state in lst_pre:
                            lst_pre[new_state].add(state)
                        else:
                            lst_pre[new_state] = {state}
        priorityQueue = util.PriorityQueue()
        for state in total_states:
            if not self.mdp.isTerminal(state):
                val_lst = []
                old_val = float("-inf")
                for action in self.mdp.getPossibleActions(state):
                    qVal = self.computeQValueFromValues(state, action)
                    if qVal > old_val:
                        old_val = qVal
                    val_lst.append(qVal)
                diff = abs(self.values[state] - old_val)
                priorityQueue.update(state, - diff)

        def get_Q_val(state):
            return max(self.getQValue(state, a) for a in self.mdp.getPossibleActions(state))
        

        for i in range(self.iterations):
            if priorityQueue.isEmpty():
                break
                
            state = priorityQueue.pop()
            if not self.mdp.isTerminal(state):
                best_action = get_Q_val(state)
                self.values[state] = best_action

            for p in lst_pre[state]:
                best_action = get_Q_val(p)
                diff = abs(self.values[p] - best_action)
                if diff > self.theta:
                    priorityQueue.update(p, -diff)
