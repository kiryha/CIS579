import networkx as nx
import matplotlib.pyplot as plt


class Graph:
    def __init__(self):
        self.graph = {}

    def add_edge(self, node, neighbour):
        if node not in self.graph:
            self.graph[node] = [neighbour]
        else:
            self.graph[node].append(neighbour)

        # For undirected graph, edge is bidirectional.
        # If it's a directed graph, remove the following three lines.
        if neighbour not in self.graph:
            self.graph[neighbour] = [node]
        else:
            self.graph[neighbour].append(node)

    def show_edges(self):
        for node in self.graph:
            for neighbour in self.graph[node]:
                print("(", node, ", ", neighbour, ")")


# # Test the graph
# graph = Graph()
# graph.add_edge((1, 0), (2, 0))
# # graph.add_edge(1, 3)
# # graph.add_edge(2, 3)
# # graph.add_edge(3, 4)
# graph.show_edges()
# print(graph.graph)

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

# Build graph
graph = {}
for state in all_states:
    graph[state] = []
    for i in range(4):
        for move in knight_moves[state[i]]:
            new_state = list(state)
            new_state[i] = move
            if tuple(new_state) in all_states:
                graph[state].append(tuple(new_state))

print(graph)