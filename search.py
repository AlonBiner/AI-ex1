"""
In search.py, you will implement generic search algorithms
"""

import util
from util import Queue
from util import PriorityQueue


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def is_goal_state(self, state):
        """
        state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()


def depth_first_search(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches
    the goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """

    solution = []  # A list of actions for the solution.
    successors = []  # This is an array that holds the layers of successors. A list of lists of tuples.
    state = problem.get_start_state()  # Initializes the initial state to the starting state of the problem.
    visited_states = set()  # Saves visited nodes in order to prevent infinite loops.

    while not problem.is_goal_state(state):
        visited_states.add(state)
        # Add to the successors array only nodes that weren't visited.
        unvisited_nodes = [node for node in problem.get_successors(state) if node[0] not in visited_states]
        successors.append(unvisited_nodes)

        # If the last layer of successors is empty, we need to back track to the last node where we can take a
        # different node.
        if not successors[-1]:
            # Remove the back tracked actions and nodes until we get to a new node:
            while not successors[-1]:
                solution.pop()
                successors.pop()
                successors[-1].pop(0)
            # Add the new node to the solution.
            state = successors[-1][0][0]
            solution.append(successors[-1][0][1])
            continue

        # Add the next node to the solution and update the current state.
        solution.append(successors[-1][0][1])
        state = successors[-1][0][0]

    # If the last state is the goal return the solution:
    if problem.is_goal_state(state):
        return solution
    # Otherwise there is no solution.
    return None
    # util.raiseNotDefined()


def breadth_first_search(problem):
    """
    Search the shallowest nodes in the search tree first.
    """
    node = problem.get_start_state()
    if problem.is_goal_state(node):
        return []
    frontier = Queue()
    frontier.push(node)
    reached = {node}
    paths_to_node = {node: []}  # This dictionary is a dictionary of the shortest paths to a node.

    while not frontier.isEmpty():
        node = frontier.pop()  # gets and removes the first node in the queue (BFS).
        path_to_father = paths_to_node[node]

        for successor in problem.get_successors(node):
            state = successor[0]
            paths_to_node[state] = path_to_father + [successor[1]]  # Updates the dictionary to the path to the
            # successor.
            if problem.is_goal_state(state):
                return paths_to_node[state]
            if state not in reached:
                reached.add(state)
                frontier.push(state)  # Add the successor to the frontier in order to expand through its path.

    # If there is no path return None:
    return None
    # util.raiseNotDefined()


def uniform_cost_search(problem):
    """
    Search the node of least total cost first.
    """
    # "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    return a_star_search(problem)


def null_heuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def a_star_search(problem, heuristic=null_heuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """
    # "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    node = problem.get_start_state()
    if problem.is_goal_state(node):
        return []

    # The priority queue wil consist of data=(node, path_to_node) priority=(action_cost, entry_time)
    frontier = PriorityQueue()
    path_to_goal = None  # Represents the solution to a goal.
    min_cost_to_goal = 0
    action_cost = 0
    solution = []  # Represents a path to a node
    # This variable is used for the priority queue.
    # If two nodes have the same action cost, the priority queue will enter
    # the node according to the first node it saw.
    entry_time = 0
    frontier.push((node, solution), (action_cost, entry_time))
    reached = set()

    while not frontier.isEmpty():
        # Get the smallest cost node:
        node, action = frontier.pop()  # Pops the smallest priority.
        path_to_node = solution + action
        path_cost = problem.get_cost_of_actions(path_to_node) + heuristic(node, problem)

        if problem.is_goal_state(node):
            # Check if the path to this goal is better:
            if path_cost < min_cost_to_goal or path_to_goal is None:
                path_to_goal = path_to_node
                min_cost_to_goal = path_cost

        elif node in reached or (path_to_goal is not None and path_cost >= min_cost_to_goal):
            # If the path costs more than the minimum goal cost, abandon this path.
            reached.add(node)
            continue

        else:
            reached.add(node)
            for successor in problem.get_successors(node):
                entry_time += 1
                frontier.push((successor[0], path_to_node + [successor[1]]), (path_cost + successor[2], entry_time))

    return path_to_goal if path_to_goal is not None else None


# Abbreviations
bfs = breadth_first_search
dfs = depth_first_search
astar = a_star_search
ucs = uniform_cost_search
