# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
 # Initialize the frontier using the start state of the problem
    frontier = util.Stack()
    start_state = problem.getStartState()
    frontier.push((start_state, []))
    # Initialize an empty set to keep track of visited states
    visited = set()

    while frontier:
        current_state, actions = frontier.pop()
        if problem.isGoalState(current_state):
            return actions
        visited.add(current_state)
        for next_state, action, cost in problem.getSuccessors(current_state):
            if next_state not in visited:
                frontier.push((next_state, actions + [action]))
    return None






def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
        #FIFO
    queue = util.Queue()
    
    #previously expanded states 
    visited = []
    
    startState = problem.getStartState()
    startNode = (startState, [], 0) #(state, action, cost)
    
    queue.push(startNode)
    
    while not queue.isEmpty():
        presentState, actions, presentCost = queue.pop()
        
        if presentState not in visited:
            #popped node state into visited list
            visited.append(presentState)

            if problem.isGoalState(presentState):
                return actions
            else:
                #lists (successor, action, stepCost)
                successors = problem.getSuccessors(presentState)
                
                for succState, succAction, succCost in successors:
                    newAction = actions + [succAction]
                    newCost = presentCost + succCost
                    newNode = (succState, newAction, newCost)

                    queue.push(newNode)
    return actions
    util.raiseNotDefined()

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"


    frontier = util.PriorityQueue()
    startLocation = problem.getStartState()
    startNode = (startLocation, [], 0)
    #(state,action,cost)
    frontier.push(startNode, 0)
    #node,priority using cost
    visitedLocation = set()

    while  frontier:
        node = frontier.pop()
        if problem.isGoalState(node[0]):
            return node[1]
        if node[0] not in visitedLocation:
            visitedLocation.add(node[0])
            for successor in problem.getSuccessors(node[0]):
                if successor[0] not in visitedLocation:
                    cost = node[2] + successor[2]
                    frontier.push((successor[0], node[1] + [successor[1]], cost), cost)

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    explored = {}
    path = []
    parents = {}
    queue = util.PriorityQueue()
    cost = {}

    # start state is known and added
    start = problem.getStartState()
    queue.push((start, 'NULL', 0), 0)

    # We don't know the direction we came in to the start node, so the direction is NULL
    explored[start] = 'NULL'

    # cost of start state
    cost[start] = 0

    # check if the start state is the goal
    if problem.isGoalState(start):
        return path

    # loop until the queue is not empty and the goal is not reached
    flag = False
    while (queue.isEmpty()==False and flag==False):

        # pop from the top of the queue
        node = queue.pop()

        # storing the path and it's direction
        explored[node[0]] = node[1]

        # check if it's goal
        if problem.isGoalState(node[0]):
            node_path = node[0]
            flag = True
            break

        # expanding the node
        for goods in problem.getSuccessors(node[0]):

            # if the successor is not explored, calculate the new cost
            if goods[0] not in explored.keys():
                concern = node[2] + goods[2] + heuristic(goods[0], problem)

                # If the cost of the successor was previously determined when growing a different node,
                # proceed if the current cost is more than the previous cost.
                if goods[0] in cost.keys():
                    if cost[goods[0]] <= concern:
                        continue

                # Push to queue, modify cost, and parent if new cost is less than old cost.
                queue.push((goods[0], goods[1], node[2] + goods[2]), concern)
                cost[goods[0]] = concern

                # successor to the store and its parent
                parents[goods[0]] = node[0]

    # discovering and storing the path
    while (node_path in parents.keys()):

        # find the parent node
        node_path_v2 = parents[node_path]

        # add a direction to the path
        path.insert(0, explored[node_path])

        # back to the previous node
        node_path = node_path_v2

    return path

    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch


