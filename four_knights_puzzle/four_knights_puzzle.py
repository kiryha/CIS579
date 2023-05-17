"""
CIS579
First Assignment: The Four Knights puzzle
Your first assignment is to write a program (in Lisp or another language) that allows you to compare the effectiveness
of the A* search algorithm to that of traditional branch and bound search.
The problem to be solved is a simplified version of the Four Knights puzzle.


# Test case.
One dimensional field of 5 cells with one knight in first cell moving to the right only.
Find a path from 1 to 5
[ * |  |  |  |  ]

# # Test Graph
# graph = Graph()
# graph.build_solution_graph_test(build_all_states_test())
# print(graph.nodes)
# path = graph.bfs(1, 5)
# print(path)


# Visualize graph:

import networkx as nx
import matplotlib.pyplot as plt

def draw_graph(graph):
    G = nx.DiGraph()  # Create a new directed graph G

    # Add nodes to the graph
    for node in graph.nodes:
        G.add_node(node)

    # Add edges to the graph
    for node, edges in graph.nodes.items():
        for edge in edges:
            G.add_edge(node, edge)

    nx.draw(G, with_labels=True)
    plt.show()

# Draw the graph
draw_graph(graph)
"""


class Graph:
    def __init__(self):
        self.nodes = {}

    # Standard data structure methods
    def add_node(self, node):
        if node not in self.nodes:
            self.nodes[node] = []

    def add_edge(self, node1, node2):
        if node1 in self.nodes and node2 in self.nodes:
            if node2 not in self.nodes[node1]:
                self.nodes[node1].append(node2)

    # Search algorithms
    def bfs(self, start, goal):
        queue = [[start]]  # Queue for the nodes to visit, [[start]] is the initial path
        visited = {start}  # Set of visited nodes
        nodes_explored = 0  # Keep track of the number of nodes explored

        while queue:
            path = queue.pop(0)  # Get the first path from the queue
            node = path[-1]  # Get the last node from the path
            nodes_explored += 1  # Increment the number of nodes explored

            # Check if we reached the goal
            if node == goal:
                print(f"BFS explored {nodes_explored} nodes.")
                return path

            # Enqueue paths with one additional move
            for next_node in self.nodes[node]:
                if next_node not in visited:
                    visited.add(next_node)
                    new_path = list(path)
                    new_path.append(next_node)
                    queue.append(new_path)

        return None  # No path found

    def branch_and_bound(self, start, goal):
        queue = [(0, [start])]  # Queue for the nodes to visit, (0, [start]) is the initial path
        visited = {start}  # Set of visited nodes
        nodes_explored = 0  # Keep track of the number of nodes explored

        while queue:
            queue.sort(key=lambda x: x[0])  # Order by path length
            path_length, path = queue.pop(0)  # Get the shortest path from the queue
            node = path[-1]  # Get the last node from the path
            nodes_explored += 1  # Increment the number of nodes explored

            # Check if we reached the goal
            if node == goal:
                print(f"Branch and Bound explored {nodes_explored} nodes.")
                return path

            # Enqueue paths with one additional move
            for next_node in self.nodes[node]:
                if next_node not in visited:
                    visited.add(next_node)
                    new_path = list(path)
                    new_path.append(next_node)
                    new_path_length = path_length + 1
                    queue.append((new_path_length, new_path))

        return None  # No path found

    def a_star(self, start, goal):

        open_set = [start]  # Nodes to be evaluated
        came_from = {}  # For each node, which node it can most efficiently be reached from
        g_score = {node: float('inf') for node in self.nodes}  # Cost of getting from start to each node
        g_score[start] = 0
        f_score = {node: float('inf') for node in
                   self.nodes}  # Estimated total cost from start to goal through each node
        f_score[start] = self.heuristic(start, goal)
        nodes_explored = 0  # Keep track of the number of nodes explored

        while open_set:
            current = min(open_set, key=f_score.get)  # Node in open_set with lowest f_score[] value
            nodes_explored += 1  # Increment the number of nodes explored

            if current == goal:
                print(f"A* explored {nodes_explored} nodes.")
                return self.reconstruct_path(came_from, current)

            open_set.remove(current)
            for neighbor in self.nodes[current]:
                tentative_g_score = g_score[current] + 1  # Each edge has a cost of 1
                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + self.heuristic(neighbor, goal)
                    if neighbor not in open_set:
                        open_set.append(neighbor)

        return None  # No path found

    def heuristic(self, state, goal):

        return sum(s != g for s, g in zip(state, goal))  # Number of knights not in goal positions

    def reconstruct_path(self, came_from, current):

        path = [current]
        while current in came_from:
            current = came_from[current]
            path.insert(0, current)

        return path

    # Build graphs for 4 Knight and Test problems
    def build_solution_graph(self, all_states):
        """
        Create a graph of all possible moves in For Knights problem
        """

        # Possible knight moves from each cell
        knight_moves = {
            1: [6, 8],
            2: [7, 9],
            3: [4, 8],
            4: [3, 9],
            5: [],
            6: [1, 7],
            7: [2, 6],
            8: [1, 3],
            9: [2, 4]
        }

        # Add all states as nodes to the graph
        for state in all_states:
            self.add_node(state)

        # Add edges to the graph
        for state in all_states:
            for i in range(4):
                for move in knight_moves[state[i]]:
                    new_state = list(state)
                    new_state[i] = move
                    new_state = tuple(new_state)
                    if new_state in all_states:
                        self.add_edge(state, new_state)

    def build_solution_graph_test(self, all_states):
        """
        Create a graph of all possible moves in test problem
        """

        # Add all states as nodes to the graph
        for state in all_states:
            self.add_node(state)

        # Add edges to the graph
        for state in all_states:
            for step in range(1, 6):  # The knight can move 1 to 5 steps
                # # Knight moves to the left
                # if state - step in all_states:
                #     self.add_edge(state, state - step)

                # Knight moves to the right
                if state + step in all_states:
                    self.add_edge(state, state + step)


def build_all_states():
    """
    Build list of all possible moves tuples in Four Knight Problem

    We use four nested loops to generate all possible combinations for the four knights
    We form a set from the positions. In a set, duplicate values are not allowed.
    If the length of the set is 4, it means all knights are on different squares.
    The knights can't land on the center square, so we check if any knight is on square 5.
    If all conditions are met, we append the state to the list of all states.
    """

    all_states = []

    for a in range(1, 10):
        for b in range(1, 10):
            for c in range(1, 10):
                for d in range(1, 10):
                    if len({a, b, c, d}) == 4:
                        if 5 not in [a, b, c, d]:
                            all_states.append((a, b, c, d))

    return all_states


def build_all_states_test():
    """
    Build list of all possible moves tuples for simple one-dimensional test case
    """

    all_states = []

    for cell in range(1, 6):
        all_states.append(cell)

    return all_states


graph = Graph()
graph.build_solution_graph(build_all_states())
start = (1, 3, 7, 9)
goal = (9, 7, 3, 1)
path = graph.bfs(start, goal)
# path = graph.a_star(start, goal)
# path = graph.branch_and_bound(start, goal)

if path is not None:
    print(path)
