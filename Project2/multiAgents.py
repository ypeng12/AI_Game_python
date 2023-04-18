# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distManhatribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and gameSuccessolstMiniMaxsor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best legalActionsList
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, current_game_state, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed gameSuccessolstMiniMaxsor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        gameSuccessolstMiniMaxsorGameState = current_game_state.generatePacmanSuccessor(action)
        newPos = gameSuccessolstMiniMaxsorGameState.getPacmanPosition()
        newFood = gameSuccessolstMiniMaxsorGameState.getFood()
        newGhostStates = gameSuccessolstMiniMaxsorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        if action == Directions.STOP:
            return -float('inf')
        #get closest food
        foodClose = None
        distManhaFoodClose = float('inf')
        for food in newFood.asList():
            disToFood = manhattanDistance(food, newPos)
            if disToFood < distManhaFoodClose:
                foodClose = food
                distManhaFoodClose = disToFood

        result = 0
        if foodClose:
            distManha = manhattanDistance(newPos, foodClose)
            result -= distManha * .25

        # ghost positions
        ghostPositions = []
        for ghostState in newGhostStates:
            ghost = ghostState.configuration.pos
            ghostPositions.append(ghost)


        DisGhostClose = float('inf')
        for ghost in ghostPositions:
            distManhatance = manhattanDistance(newPos, ghost)
            if distManhatance < DisGhostClose:
                DisGhostClose = distManhatance
        if DisGhostClose <= 3:
            result -= (3 - DisGhostClose) * 1000

        result += gameSuccessolstMiniMaxsorGameState.data.score

        if newPos == current_game_state.getPacmanPosition():
            result -= 1
        return result

def scoreEvaluationFunction(current_game_state):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one distManhaplayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return current_game_state.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """
    def getResult(self, gameState, depth):
        if depth == self.depth * gameState.getNumAgents() or gameState.isWin() or gameState.isLose():
            return (None, self.evaluationFunction(gameState))
        if depth % gameState.getNumAgents() == 0:
            # pacman
            return self.getMax(gameState, depth)
        else:
            # ghosts
            return self.getMin(gameState, depth)

    def getMin(self, gameState, depth):
        listLegalAction = gameState.getLegalActions(depth % gameState.getNumAgents()) or gameState.isWin() or gameState.isLose() 
        if len(listLegalAction) == 0:
            return (None, self.evaluationFunction(gameState))

        minNum = (None, float("inf"))
        for action in listLegalAction:
            gameSuccessor = gameState.generateSuccessor(depth % gameState.getNumAgents(), action)
            lstMiniMax = self.getResult(gameSuccessor, depth+1)
            if lstMiniMax[1] < minNum[1]:
                minNum = (action, lstMiniMax[1])
        return minNum

    def getMax(self, gameState, depth):
        listLegalAction = gameState.getLegalActions(0) or gameState.isWin() or gameState.isLose() 
        if len(listLegalAction) == 0:
            return (None, self.evaluationFunction(gameState))

        maxNum = (None, -float("inf"))
        for action in listLegalAction:
            gameSuccessor = gameState.generateSuccessor(0, action)
            lstMiniMax = self.getResult(gameSuccessor, depth+1)
            if lstMiniMax[1] > maxNum[1]:
                maxNum = (action, lstMiniMax[1])
        return maxNum
    
    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal listLegalAction for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the gameSuccessolstMiniMaxsor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the result number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        lstMiniMax = self.getResult(gameState, 0)
        return lstMiniMax[0]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        result = self.alpha_beta_search(gameState, 0, -float("inf"), float("inf"))
        return result[0]

    def alpha_beta_search(self, gameState, depth, alpha, beta):
        if depth == self.depth * gameState.getNumAgents() or gameState.isWin() or gameState.isLose():
            return (None, self.evaluationFunction(gameState))
        elif depth % gameState.getNumAgents() == 0:
            # Pacman's turn
            return self.alpha_value(gameState, depth, alpha, beta)
        else:
            # Ghosts' turn
            return self.beta_value(gameState, depth, alpha, beta)

    def alpha_value(self, gameState, depth, alpha, beta):
        legalActions = gameState.getLegalActions(0)
        if not legalActions or depth == self.depth * gameState.getNumAgents() or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        bestAction, bestScore = None, -float("inf")
        for action in legalActions:
            successorState = gameState.generateSuccessor(0, action)
            score = self.alpha_beta_search(successorState, depth+1, alpha, beta)[1]
            if score > bestScore:
                bestAction, bestScore = action, score
            if bestScore > beta:
                return (bestAction, bestScore)
            alpha = max(alpha, bestScore)
        return (bestAction, bestScore)


        
    def beta_value(self, gameState, depth, alpha, beta):
        legalActions = gameState.getLegalActions(depth % gameState.getNumAgents())
        if not legalActions or depth == self.depth * gameState.getNumAgents() or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        bestAction, bestScore = None, float("inf")
        for action in legalActions:
            successorState = gameState.generateSuccessor(depth % gameState.getNumAgents(), action)
            score = self.alpha_beta_search(successorState, depth+1, alpha, beta)[1]
            if score < bestScore:
                bestAction, bestScore = action, score
            if bestScore < alpha:
                return (bestAction, bestScore)
            beta = min(beta, bestScore)
        return (bestAction, bestScore)
class ExpectimaxAgent(MultiAgentSearchAgent):    

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction
        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        result = self.expectimax_search(gameState, 0)
        return result[0]

    def expectimax_search(self, gameState, depth):
        if depth == self.depth * gameState.getNumAgents() or gameState.isWin() or gameState.isLose():
            return (None, self.evaluationFunction(gameState))
        if depth % gameState.getNumAgents() == 0:
            # Pacman's turn
            return self.max_value(gameState, depth)
        else:
            # Ghosts' turn
            return self.exp_value(gameState, depth)

    def exp_value(self, gameState, depth):
        legalActions = gameState.getLegalActions(depth % gameState.getNumAgents())
        if not legalActions:
            return (None, self.evaluationFunction(gameState))

        probability = 1.0 / len(legalActions)
        expectedValue = 0.0
        for action in legalActions:
            successorState = gameState.generateSuccessor(depth % gameState.getNumAgents(), action)
            result = self.expectimax_search(successorState, depth + 1)
            expectedValue += result[1] * probability
        return (None, expectedValue)

    def max_value(self, gameState, depth):
        legalActions = gameState.getLegalActions(0)
        if not legalActions:
            return (None, self.evaluationFunction(gameState))

        bestAction, bestScore = None, -float("inf")
        for action in legalActions:
            successorState = gameState.generateSuccessor(0, action)
            result = self.expectimax_search(successorState, depth + 1)
            if result[1] > bestScore:
                bestAction, bestScore = action, result[1]
        return (bestAction, bestScore)

def betterEvaluationFunction(current_game_state):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).
    DESCRIPTION: chase after ghost if the manhattan distance to it is < 10
                  and the ghost is scared.
                  go towards closest foods.
                  run away from ghosts <= 3 distance away
    """
    pac_pos = current_game_state.getPacmanPosition()
    food_grid = current_game_state.getFood()
    ghost_states = current_game_state.getGhostStates()
    scared_counts = [ghostState.scaredTimer for ghostState in ghost_states]

    # find closest food
    nearby_food = None
    nearby_food_dist = float('inf')
    for food in food_grid.asList():
        dist_to_food = manhattanDistance(food, pac_pos)
        if dist_to_food < nearby_food_dist:
            nearby_food = food
            nearby_food_dist = dist_to_food

    total_score = 0
    if nearby_food:
        dist_to_nearby_food = manhattanDistance(pac_pos, nearby_food)
        total_score -= dist_to_nearby_food * .25

    # find distance to closest ghost
    ghost_positions = []
    for ghost_state in ghost_states:
        ghost_pos = ghost_state.configuration.pos
        ghost_positions.append(ghost_pos)

    num_of_ghost_dis = scared_counts.index(max(scared_counts))
    run_away_ghost = scared_counts[num_of_ghost_dis]
    ghost_dist = float('inf')
    for ghost in ghost_positions:
        dist = manhattanDistance(pac_pos, ghost)
        if dist < ghost_dist:
            ghost_dist = dist

    # run away from ghosts <= 3 distance away or chase scared ghosts < 10 distance away
    if not run_away_ghost and ghost_dist <= 3:
        total_score -= (3 - ghost_dist) * 1000
    else:
        for count in scared_counts:
            scared_ghost_pos = ghost_states[scared_counts.index(count)].configuration.pos
            dist_to_scared_ghost = manhattanDistance(pac_pos, scared_ghost_pos)
            if count > 0 and dist_to_scared_ghost < 10:
                total_score += dist_to_scared_ghost

    # add current score
    total_score += current_game_state.data.score

    # avoid standing still
    if pac_pos == current_game_state.getPacmanPosition():
        total_score -= 1

    return total_score
# Abbreviation
better = betterEvaluationFunction