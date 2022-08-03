'''Dijkstra and A* multi-AGV routing algorithm, with avoiding conflicts '''

import sys
import pandas as pd
import numpy as np

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

def h(n):
    H={}
    for i in range(0,159):
        H[i] =1
    return H[n]

def a_star_algorithm(graph,start, stop,speed,TaskID,endtime_lasttask):
    # In this open_lst is a list of nodes which have not been visited, but who's
    # neighbours haven't all been always inspected, It starts off with the start node
    # And closed_lst is a list of nodes which have been visited
    # and who's neighbors have been always inspected
    open_lst = set([start])
    closed_lst = set([])

    arrive_time_lst = {}
    for id in range(0,500):
        arrive_time_lst[id] = {}



    # dis_start has present distances from start to all other nodes
    # the default value is +infinity
    dis_start = {}
    dis_start[start] = 0

    # adjac_node contains an adjac mapping of all nodes
    adjac_node = {}
    adjac_node[start] = start

    while len(open_lst) > 0:
        n = None
        # it will find a node with the lowest value of f() -
        for v in open_lst:
            if n == None or dis_start[v] + h(v) < dis_start[n] + h(n):
                n = v
        if n == None:
            print('Path does not exist!')
            return None
        # if the current node is the stop
        # then we start again from start
        if n == stop:
            reconst_path = []
            while adjac_node[n] != n:
                reconst_path.append(n)
                n = adjac_node[n]

            reconst_path.append(start)
            reconst_path.reverse()

            if n == start:
                arrive_time_lst[TaskID][n]=2/speed + endtime_lasttask #previous route end time

            for i in range(1,len(reconst_path)):
                arrive_time_lst[TaskID][reconst_path[i]] = 2 / speed + arrive_time_lst[TaskID][reconst_path[i-1]]

            completion_time = arrive_time_lst[TaskID][reconst_path[len(reconst_path)-1]]

            print('Path found: {}'.format(reconst_path))
            print(arrive_time_lst[TaskID])
            #return reconst_path,arrive_time_lst[TaskID]
            return reconst_path, completion_time

        # for all the neighbors of the current node do
        for m in graph.get_outgoing_edges(n):
            # if the current node is not presentin both open_lst and closed_lst
            # add it to open_lst and note n as it's par
            if m not in open_lst and m not in closed_lst:
                open_lst.add(m)
                adjac_node[m] = n
                dis_start[m] = dis_start[n] + graph.value(n,m)

            # otherwise, check if it's quicker to first visit n, then m
            # and if it is, update par data and poo data
            # and if the node was in the closed_lst, move it to open_lst
            else:
                if dis_start[m] > dis_start[n] + graph.value(n,m):
                    dis_start[m] = dis_start[n] + graph.value(n,m)
                    adjac_node[m] = n

                    if m in closed_lst:
                        closed_lst.remove(m)
                        open_lst.add(m)

        # remove n from the open_lst, and add it to closed_lst
        # because all of his neighbors were inspected
        open_lst.remove(n)
        closed_lst.add(n)

    print('Path does not exist!')
    return None

#def avoid_conflicts():


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

    #Route planning
    arrive_time = {}
    arrive_time_back = {}
    for vehicle in range(1,7):
        arrive_time[vehicle] = {}
        arrive_time_back[vehicle] = {}
    for vehicle in range(1,7):
        for id in range(-1,150):
            arrive_time[vehicle][id] = {}
            arrive_time_back[vehicle][id] ={}

    # Input task list
    #Task_list = pd.read_excel('Task_list_Test.xlsx', sheet_name='Tasklist', usecols="A:G", skiprows=0, nrows=6, dtype=object)
    Task_list = pd.read_excel('Task_list_Test.xlsx', sheet_name='Tasklist', usecols=[0,3,6,7,8,9,10], skiprows=0, nrows=7,
                              dtype=object)
    #Task_list_visibility = pd.read_excel('Task_list.xlsx', sheet_name='Tasklist', usecols=[0,3,6,7,8,9,10], skiprows=0, nrows=141,
                              #dtype=object)

    Task_list_Forklift = Task_list.loc[Task_list['Task type'] == 'C04_CMD']
    Task_list_AGV = Task_list.loc[Task_list['Task type'] != 'C04_CMD']
    #Tasks = Task_list_visibility['Task ID'].tolist()  # the task ID


    reconst_path, arrive_time[1][0] = a_star_algorithm(graph, Task_list.loc[0, 'Start node'], Task_list.loc[0, 'End node'],
                                                 speed=1, TaskID=0, endtime_lasttask=0)
    reconst_path_back, arrive_time_back[1][0] =a_star_algorithm(graph, Task_list.loc[0, 'End node'], Task_list.loc[1, 'Start node'], speed=1, TaskID=0,
                     endtime_lasttask=arrive_time[1][0][Task_list.loc[0, 'End node']])

    for i in range(1,6):
        print('---------Task ID',i,'--------------')
        reconst_path,arrive_time[i+1][i]=a_star_algorithm(graph,Task_list.loc[i,'Start node'],Task_list.loc[i,'End node'],speed=1,TaskID=i,endtime_lasttask=arrive_time_back[i+1][i-1][Task_list.loc[i, 'Start node']])
        reconst_path_back, arrive_time_back[i+1][i]=a_star_algorithm(graph, Task_list.loc[i, 'End node'], Task_list.loc[i+1, 'Start node'], speed=1, TaskID=i,endtime_lasttask=arrive_time[i+1][i][Task_list.loc[i, 'End node']])


