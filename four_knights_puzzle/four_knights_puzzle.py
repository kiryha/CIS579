"""

Visualize graph:

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

    def build_solution_graph(self, all_states):
        """
        Create a graph for all possible moves in For Knights problem
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

    def add_node(self, node):
        if node not in self.nodes:
            self.nodes[node] = []

    def add_edge(self, node1, node2):
        if node1 in self.nodes and node2 in self.nodes:
            if node2 not in self.nodes[node1]:
                self.nodes[node1].append(node2)

    def bfs(self, start, goal):
        queue = [[start]]  # Queue for the nodes to visit, [[start]] is the initial path
        visited = {start}  # Set of visited nodes

        while queue:
            path = queue.pop(0)  # Get the first path from the queue
            node = path[-1]  # Get the last node from the path

            # Check if we reached the goal
            if node == goal:
                return path

            # Enqueue paths with one additional move
            for next_node in self.nodes[node]:
                if next_node not in visited:
                    visited.add(next_node)
                    new_path = list(path)
                    new_path.append(next_node)
                    queue.append(new_path)

        return None  # No path found


def build_all_states():

    all_states = []  # List to store all states

    # We use four nested loops to generate all possible combinations for the four knights
    for a in range(1, 10):
        for b in range(1, 10):
            for c in range(1, 10):
                for d in range(1, 10):

                    # We form a set from the positions. In a set, duplicate values are not allowed.
                    # If the length of the set is 4, it means all knights are on different squares.
                    if len({a, b, c, d}) == 4:

                        # The knights can't land on the center square, so we check if any knight is on square 5.
                        if 5 not in [a, b, c, d]:
                            # If all conditions are met, we append the state to the list of all states.
                            all_states.append((a, b, c, d))

    return all_states


graph = Graph()
graph.build_solution_graph(build_all_states())
# print(graph.nodes)


start = (1, 3, 7, 9)
goal = (9, 7, 3, 1)
path = graph.bfs(start, goal)

if path is not None:
    print("Found a path:", path)
else:
    print("No path found.")