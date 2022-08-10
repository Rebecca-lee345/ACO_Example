# -*- coding: utf-8 -*-
import random
import copy
import time
import sys
import math
import pandas as pd
#Task list input
Task_list = pd.read_excel('Task_list_ACO_Test.xlsx', sheet_name='Tasklist', usecols=[0,3,6,7,8], skiprows=0, nrows=7,
                              dtype=object)

# parameters
(ALPHA, BETA, RHO, Q) = (1.0,2.0,0.5,100.0)

(node_num, ant_num) = (25,6)

#visibility graph---the distance between nodes  and pheromone graph
visibility_graph = [ [0.0 for col in range(node_num)] for raw in range(node_num)]
pheromone_graph = [ [1.0 for col in range(node_num)] for raw in range(node_num)]

#Roadmap
Manufacturing_Graph = pd.read_excel('Graph.xlsx', sheet_name='Sheet2', usecols="A:C", skiprows=0, nrows=27,dtype=object)
nodes = list(range(0, 25))

Node1 = Manufacturing_Graph['Node1'].tolist()
Node2 = Manufacturing_Graph['Node2'].tolist()

init_graph = {}
for node in nodes:
    init_graph[node] = {}

for i in range(0, 27):
    init_graph[Node1[i]][Node2[i]] = Manufacturing_Graph.loc[i,'Distance']



#-----------graph-------------------
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

#----------- Ant -----------
class Ant(object):
    # 初始化
    def __init__(self,ID):
        self.ID = ID                 # ID
        self.__clean_data()          # 随机初始化出生点
    # 初始数据
    def __clean_data(self):
        self.path = []               # 当前蚂蚁的路径
        self.total_distance = 0.0    # 当前路径的总距离
        self.move_count = 0          # 移动次数
        self.current_node = -1       # 当前停留的城市
        self.open_table_city = [True for i in range(node_num)] # 探索城市的状态
        node_index = random.randint(0,node_num-1) # 随机初始出生点
        self.current_node = node_index
        self.path.append(node_index)
        self.open_table_city[node_index] = False
        self.move_count = 1


    def __cal_total_distance(self):
        temp_distance = 0.0
        for i in range(1, node_num):
            start, end = self.path[i], self.path[i-1]
            temp_distance += visibility_graph[start][end]
        # 回路
        end = self.path[0]
        temp_distance += visibility_graph[start][end]
        self.total_distance = temp_distance






#----------- ant colony routing -----------
def ACO_Routing_test(graph):
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
    start_node=Task_list.loc[0,'Start node']
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

    path = []
    target_node = Task_list.loc[0,'End node']
    node = target_node

    while node != start_node:
        path.append(node)
        node = previous_nodes[node]

    # Add the start node manually
    path.append(start_node)

    print("We found the following best path with a value of {}.".format(shortest_path[target_node]))
    print(" -> ",'Route' ,list(reversed(path)))

    return previous_nodes, shortest_path



graph_input = Graph(nodes, init_graph)
# neighbors = graph_input.get_outgoing_edges(8)
# dic={}
# for i in neighbors:
#     dic[i]={}
#
# dic[9]=6


class ACO_Routing(object):
    def __init__(self, n = node_num,graph=graph_input):

        # initial node number is n
        self.n = n
        self.new()
        # calculate the distance between nodes
        for i in range(node_num):
            for j in range(node_num):
                temp_distance = random.randint(225, 450)
                visibility_graph[i][j] = float(int(temp_distance + 0.5))
        self.search_path(graph)


    # 初始化
    def new(self):
        self.__running = False
        # 初始城市之间的距离和信息素
        for i in range(node_num):
            for j in range(node_num):
                pheromone_graph[i][j] = 1.0
        self.ants = [Ant(ID) for ID in range(ant_num)]  # 初始蚁群
        self.best_ant = Ant(-1)                          # 初始最优解
        self.best_ant.total_distance = 1 << 31           # 初始最大距离
        self.iter = 1                                    # 初始化迭代次数
        for ant in self.ants:
            ant.path = []
            ant.current_node = Task_list.loc[0,'Start node']
            ant.total_distance = 0
            ant.path.append(ant.current_node)



    def search_path(self, graph):
        # 开启线程
        self.__running = True

        while self.__running:
            print('--------------iter',self.iter)
            # go through each ant
            for ant in self.ants:
                # search a path
                target_node= Task_list.loc[0,'End node']

                while ant.current_node != target_node:
                    next_node = -1
                    # print(u"迭代次数：", self.iter, ant.path)
                    neighbors = graph.get_outgoing_edges(ant.current_node)
                    #select_nodes_prob = [0.0 for i in neighbors]  # store the probability of choosing next task
                    select_nodes_prob={}
                    for i in neighbors:
                        select_nodes_prob[i]={}
                    total_prob = 0.0

                    # search next node
                    #print(ant.current_node)
                    for i in neighbors:
                        try:
                            # Calculate the probability
                            select_nodes_prob[i] = pow(pheromone_graph[ant.current_node][i],ALPHA) * pow((1.0 / visibility_graph[ant.current_node][i]), BETA)
                            total_prob += select_nodes_prob[i]

                        except ZeroDivisionError as e:
                            print('Ant ID: {ID}, current task: {current}, target task: {target}'.format(
                                ID=ant.ID, current=ant.current_node, target=i))
                            sys.exit(1)
                        # 轮盘堵选择node
                    if total_prob > 0.0:
                        # generate a random probability,0.0-total_prob
                        temp_prob = random.uniform(0.0, total_prob)
                        for i in neighbors:
                            # 轮次相减
                            temp_prob -= select_nodes_prob[i]
                            print(i)
                            print(temp_prob)
                            if temp_prob < 0.0:
                                next_node = i
                                ant.total_distance += init_graph[ant.current_node][next_node]
                                ant.current_node=next_node
                                ant.path.append(next_node)

                                break
                            else:
                                continue
                # 与当前最优蚂蚁比较
                if ant.total_distance < self.best_ant.total_distance:
                    # 更新最优解
                    self.best_ant = copy.deepcopy(ant)
            # 更新信息素
            self.__update_pheromone_gragh()
            print (u"迭代次数：",self.iter,u"最佳路径总距离：",int(self.best_ant.total_distance))
            print(u"the index of iteration：", self.iter, ant.path, ant.current_node)

            self.iter += 1
            if self.iter == 50:
                break
    # 更新信息素
    def __update_pheromone_gragh(self):
        # obtain the pheromone of each ant at the route
        temp_pheromone = [[0.0 for col in nodes] for raw in nodes]
        for ant in self.ants:
            for i in range(1, len(ant.path)):
                start, end = ant.path[i - 1], ant.path[i]
                # 在路径上的每两个相邻城市间留下信息素，与路径总距离反比
                temp_pheromone[start][end] += Q / self.best_ant.total_distance
                temp_pheromone[end][start] = temp_pheromone[start][end]
        # update all the pheromone，old pheromone decrease with the addition new pheromone
        for i in nodes:
            for j in nodes:
                pheromone_graph[i][j] = pheromone_graph[i][j] * RHO + temp_pheromone[i][j] * ALPHA

#----------- 程序的入口处 -----------
if __name__ == '__main__':

    ACO_Routing()
