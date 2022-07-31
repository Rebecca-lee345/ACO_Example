'''Dijkstra multi-AGV routing algorithm, avoiding conflicts '''

import sys
import pandas as pd

class Graph(object):
    def __init__(self, nodes, init_graph):
        self.nodes = nodes
        self.graph = self.construct_graph(nodes, init_graph)

    def construct_graph(self, nodes, init_graph):
        '''
        This method makes sure that the graph is symmetrical. In other words, if there's a path from node A to B with a value V, there needs to be a path from node B to node A with a value V.
        '''
        graph = {}
        for node in nodes:
            graph[node] = {}

        graph.update(init_graph)

        for node, edges in graph.items():
            for adjacent_node, value in edges.items():
                if graph[adjacent_node].get(node, False) == False:
                    graph[adjacent_node][node] = value

        return graph

    def get_nodes(self):
        "Returns the nodes of the graph."
        return self.nodes

    def get_outgoing_edges(self, node):
        "Returns the neighbors of a node."
        connections = []
        for out_node in self.nodes:
            if self.graph[node].get(out_node, False) != False:
                connections.append(out_node)
        return connections

    def value(self, node1, node2):
        "Returns the value of an edge between two nodes."
        return self.graph[node1][node2]


def dijkstra_algorithm(graph, start_node):
    unvisited_nodes = list(graph.get_nodes())

    # We'll use this dict to save the cost of visiting each node and update it as we move along the graph
    shortest_path = {}

    # We'll use this dict to save the shortest known path to a node found so far
    previous_nodes = {}

    # We'll use max_value to initialize the "infinity" value of the unvisited nodes
    max_value = sys.maxsize
    for node in unvisited_nodes:
        shortest_path[node] = max_value
    # However, we initialize the starting node's value with 0
    shortest_path[start_node] = 0

    # The algorithm executes until we visit all nodes
    while unvisited_nodes:
        # The code block below finds the node with the lowest score
        current_min_node = None
        for node in unvisited_nodes:  # Iterate over the nodes
            if current_min_node == None:
                current_min_node = node
            elif shortest_path[node] < shortest_path[current_min_node]:
                current_min_node = node

        # The code block below retrieves the current node's neighbors and updates their distances
        neighbors = graph.get_outgoing_edges(current_min_node)
        for neighbor in neighbors:
            tentative_value = shortest_path[current_min_node] + graph.value(current_min_node, neighbor)
            if tentative_value < shortest_path[neighbor]:
                shortest_path[neighbor] = tentative_value
                # We also update the best path to the current node
                previous_nodes[neighbor] = current_min_node

        # After visiting its neighbors, we mark the node as "visited"
        unvisited_nodes.remove(current_min_node)

    return previous_nodes, shortest_path


def generate_result(previous_nodes, shortest_path, start_node, target_node):
    path = []
    node = target_node

    while node != start_node:
        path.append(node)
        node = previous_nodes[node]

    # Add the start node manually
    path.append(start_node)

    print("We found the following best path with a value of {}.".format(shortest_path[target_node]))
    print(" -> ",'Route' ,list(reversed(path)))




# ----------- main -----------
if __name__ == '__main__':
    #input road map
    Manufacturing_Graph = pd.read_excel('Graph.xlsx', sheet_name='Sheet1', usecols="A:C", skiprows=0, nrows=162,dtype=object)
    nodes = list(range(0,159))

    Node1 = Manufacturing_Graph['Node1'].tolist()
    Node2 = Manufacturing_Graph['Node2'].tolist()

    init_graph = {}
    for node in nodes:
        init_graph[node] = {}

    for i in range(0,161):
        init_graph[Node1[i]][Node2[i]] = 2

    graph = Graph(nodes, init_graph)

    # Input task list
    #Task_list = pd.read_excel('Task_list_Test.xlsx', sheet_name='Tasklist', usecols="A:G", skiprows=0, nrows=6, dtype=object)
    Task_list = pd.read_excel('Task_list_Test.xlsx', sheet_name='Tasklist', usecols=[0,3,6,7,8], skiprows=0, nrows=6,
                              dtype=object)

    Task_list_Forklift = Task_list.loc[Task_list['Task type'] == 'C04_CMD']
    Task_list_AGV = Task_list.loc[Task_list['Task type'] != 'C04_CMD']
    Tasks = Task_list['Task ID'].tolist()  # the task ID

    for i in Tasks:
        previous_nodes, shortest_path = dijkstra_algorithm(graph=graph, start_node=Task_list.loc[i,'Start node'])
        generate_result(previous_nodes, shortest_path, start_node=Task_list.loc[i,'Start node'], target_node=Task_list.loc[i,'End node'])

    #previous_nodes, shortest_path = dijkstra_algorithm(graph=graph, start_node=57)
    #generate_result(previous_nodes, shortest_path, start_node=57, target_node=122)
