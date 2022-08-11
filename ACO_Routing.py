# -*- coding: utf-8 -*-
import random
import copy
import time
import sys
import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Task list input
Task_list = pd.read_excel('Task_list_ACO.xlsx', sheet_name='Tasklist_routing', index_col=0,usecols=[0,3,6,7,8,9,10,11], skiprows=0, nrows=142,
                              dtype=object)

# parameters
(ALPHA, BETA, RHO, Q) = (1.0,2.0,0.5,100.0)

(node_num, ant_num) = (25,40)

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



#-----------graph: roadmap-------------------
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

graph_input = Graph(nodes, init_graph)
# neighbors = graph_input.get_outgoing_edges(8)
# dic={}
# for i in neighbors:
#     dic[i]={}
#
# dic[9]=6

#scheduling_result=[[1, 11, 8],[4, 2, 3],[7, 9, 6],[10, 0, 5],[13, 17, 14],[16, 12, 15]]
scheduling_result=[[1, 0, 43, 44, 116, 63, 20, 51, 23, 22, 119, 83, 50, 86, -1, 18, -1, -1, 40, 64, 21, 39, 85, -1, -1, 82, -1, 95, -1, 52, -1, 93, -1, 55]
,[4, -1, -1, 46, 6, 9, 84, 49, 5, 8, 80, -1, 120, 90, 121, 38, 91, 25, 115, 125, 118, -1, -1, -1, -1, 96, -1, -1, -1, -1, -1, -1, -1]
,[7, -1, -1, -1, -1, 79, 2, 24, -1, -1, -1, 62, -1, -1, 87, 3, 45, 81, 89, 48, -1, -1, 94, -1, 92, 47, -1, -1, -1, -1, -1, -1, -1]
,[10, 114, 117, 123, -1, 19, 124, 54, 11, 41, 88, 53, 126, 127, -1, -1, -1, 122, 42, -1, -1, 128, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
,[13, 27, 36, 110, 134, 15, 56, 74, 137, 12, 135, 28, 103, 97, 102, 61, 107, 59, 31, 130, 73, 57, 105, 129, 65, 76, 108, 101, 71, 68, 58, 133, 17]
,[16, 99, 136, 66, 98, 72, 111, 77, 139, 67, 131, 112, 78, 69, 100, 70, 138, 33, 132, 75, 26, 32, 104, 34, 106, 37, 60, 30, 14, 113, 35, 109, 29]]

class ACO_Routing(object):
    def __init__(self, n = node_num,graph=graph_input, scheudling_list=scheduling_result):

        # initial node number is n
        self.n = n
        #self.new()
        # calculate the distance between nodes
        for i in range(node_num):
            for j in range(node_num):
                temp_distance = random.randint(225, 450)
                visibility_graph[i][j] = float(int(temp_distance + 0.5))
        #search the path
        self.search_path(graph,scheudling_list)


    # 初始化
    def new(self):
        #self.__running = False
        # 初始城市之间的距离和信息素
        for i in range(node_num):
            for j in range(node_num):
                pheromone_graph[i][j] = 1.0
        self.ants = [Ant(ID) for ID in range(ant_num)]  # initial ant number
        self.best_ant = Ant(-1)                          # initial best ant
        self.best_ant.total_distance = 1 << 31           # initial maximum distance for each ant
        for ant in self.ants:
            ant.path = []
            ant.total_distance = 0

    def __clean_data(self):
        # create a dic to store the best path for each vehicle each task
        vehicle_best_path = {}
        for vehicle in range(6):
            vehicle_best_path[vehicle] = {}
        for vehicle in range(6):
            for task_id in range(-1, 150):
                vehicle_best_path[vehicle][task_id] = {}




    def search_path(self, graph, scheudling_list):
        # 开启线程
        self.__running = True
        self.iter = 1
        self.iter_total_completion_time=[]


        while self.__running:
            # create a dic to store the best path for each vehicle each task
            vehicle_best_path = {}
            for vehicle in range(6):
                vehicle_best_path[vehicle] = {}
            for vehicle in range(6):
                for task_id in range(-1, 150):
                    vehicle_best_path[vehicle][task_id] = {}


            #creat the list to store the path travelling time
            total_completion_time = [0.0 for raw in range(6)]
            #create the list to store the task finish time at each vehicle
            task_completion_time={}
            for vehicle in range(6):
                task_completion_time[vehicle]={}
            for vehicle in range(6):
                for task_id in range(-1,150):
                    task_completion_time[vehicle][task_id] = {}




            print('--------------iter',self.iter)
            for task_location in range(len(scheudling_list[0])-2):
                for vehicle in range(6):
                    if scheudling_list[vehicle][task_location] != -1:
                        self.new()
                        # go through each ant for the forward path searching
                        for ant in self.ants:
                            # search a path
                            ant.current_node=Task_list.loc[scheudling_list[vehicle][task_location],'Start node']
                            ant.path.append(ant.current_node)
                            target_node= Task_list.loc[scheudling_list[vehicle][task_location],'End node']

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

                                        #print(i,temp_prob)
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
                        #print (u"vehicle number：",vehicle,u"最佳路径总距离：",int(self.best_ant.total_distance))
                        #print(u"the index of iteration：", self.iter, self.best_ant.path, ant.current_node)
                        # store the best path for each vehicle each task for the forward path searching
                        vehicle_best_path[vehicle][scheudling_list[vehicle][task_location]] =self.best_ant.path
                        total_completion_time[vehicle] += self.best_ant.total_distance/Task_list.loc[scheudling_list[vehicle][task_location],'speed']+Task_list.loc[scheudling_list[vehicle][task_location],'loading time']

                        #-------------------------------------------------------------------------
                        #backward path searching for each vehicle each task
                        self.new()
                        # go through each ant for the back path searching
                        for ant in self.ants:
                            # search a path
                            ant.current_node = Task_list.loc[scheudling_list[vehicle][task_location], 'End node']
                            ant.path.append(ant.current_node)
                            target_node = Task_list.loc[scheudling_list[vehicle][task_location+1], 'Start node']

                            while ant.current_node != target_node:
                                next_node = -1
                                # print(u"迭代次数：", self.iter, ant.path)
                                neighbors = graph.get_outgoing_edges(ant.current_node)
                                # select_nodes_prob = [0.0 for i in neighbors]  # store the probability of choosing next task
                                select_nodes_prob = {}
                                for i in neighbors:
                                    select_nodes_prob[i] = {}
                                total_prob = 0.0

                                # search next node
                                # print(ant.current_node)
                                for i in neighbors:
                                    try:
                                        # Calculate the probability
                                        select_nodes_prob[i] = pow(pheromone_graph[ant.current_node][i], ALPHA) * pow(
                                            (1.0 / visibility_graph[ant.current_node][i]), BETA)
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

                                        # print(i,temp_prob)
                                        if temp_prob < 0.0:
                                            next_node = i
                                            ant.total_distance += init_graph[ant.current_node][next_node]
                                            ant.current_node = next_node
                                            ant.path.append(next_node)

                                            break
                                        else:
                                            continue
                            # compare it with the best ant
                            if ant.total_distance < self.best_ant.total_distance:
                                # update the best solution
                                self.best_ant = copy.deepcopy(ant)
                        # update pheromone
                        self.__update_pheromone_gragh()
                        #print(u"vehicle number：", vehicle, u"最佳路径总距离：", int(self.best_ant.total_distance))
                        #print(u"the index of iteration：", self.iter, self.best_ant.path, ant.current_node)
                        # store the best path for each vehicle each task for the forward path searching
                        vehicle_best_path[vehicle][scheudling_list[vehicle][task_location]].extend(self.best_ant.path)

                        total_completion_time[vehicle] += self.best_ant.total_distance/Task_list.loc[scheudling_list[vehicle][task_location],'speed']+Task_list.loc[scheudling_list[vehicle][task_location],'unloading time']
                        task_completion_time[vehicle][scheudling_list[vehicle][task_location]] = total_completion_time[vehicle]
                    else:
                        idle_time = random.randint(180, 360)
                        total_completion_time[vehicle]+=idle_time
            self.iter_total_completion_time.append(max(total_completion_time))
            self.iter += 1
            if self.iter == 100:
                print(vehicle_best_path)
                print(total_completion_time)
                print(self.iter_total_completion_time)
                return self.iter_total_completion_time

                self.__running= False
            else:
                continue
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

def iteration_visulazation(iter,total_completion_time):
    # Apply the default theme
    sns.set()
    x = range(0, iter - 1)
    y = total_completion_time
    plt.plot(x, y)
    plt.xlabel('The iteration')
    plt.ylabel('The makespan of the operation')
    plt.show()

#----------- 程序的入口处 -----------
if __name__ == '__main__':

    Routing=ACO_Routing()
    total_completion_time_result=Routing.iter_total_completion_time
    iteration_visulazation(100, total_completion_time_result)
