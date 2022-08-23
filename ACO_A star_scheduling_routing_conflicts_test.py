#A* and ACO scheduling test with tackling conflicts


# -*- coding: utf-8 -*-
import random
import copy
import time
import sys
import math
import tkinter  # //GUI模块
import threading
from functools import reduce
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# 参数
'''
ALPHA:信息启发因子，值越大，则蚂蚁选择之前走过的路径可能性就越大
      ，值越小，则蚁群搜索范围就会减少，容易陷入局部最优
BETA:Beta值越大，蚁群越就容易选择局部较短路径，这时算法收敛速度会
     加快，但是随机性不高，容易得到局部的相对最优
'''
(ALPHA, BETA, RHO, Q) = (1.0, 2.0, 0.5, 100.0)
# ----Ant----
Ant_num = 6

# Input task list
Task_list = pd.read_excel('Task_list.xlsx', sheet_name='Tasklist', usecols=[0,3,6,7,8,9,10,11], skiprows=0, nrows=141, dtype=object)
Task_list_routing = pd.read_excel('Task_list.xlsx', sheet_name='Tasklist_routing', index_col=0,usecols=[0,3,6,7,8,9,10,11], skiprows=0, nrows=142, dtype=object)
Task_list_Forklift = Task_list.loc[Task_list['Task type'] == 'C04_CMD']
Task_list_AGV = Task_list.loc[Task_list['Task type'] != 'C04_CMD']
# input visibility graph
visibility_graph=pd.read_excel('Visibility_graph.xlsx', sheet_name='Sheet1', usecols="B:EK", skiprows=0, nrows=141, dtype=object)
visibility_graph=visibility_graph.values



# Parameters definition
#Line = list(range(0, 3))  # the number of lines in the roadmap
#Stop = list(range(0, 450))  # the number of stops in each line
Forklifts = list(range(0, 2))  # the number of forklifts
AGVs = list(range(0, 4))  # the number of AGVs
Tasks = Task_list['Task ID'].tolist()  # the task ID
#H = list(range(0, 600))  # the avaulable time (s), for now 1 hour

# pheromone matrix & Ant visibility matrix
pheromone_graph = [[1.0 for col in Tasks] for raw in Tasks]
#visibility_graph = [[0.0 for col in Tasks] for raw in Tasks]


# -----Ants-----
class Ant(object):
    # Initial
    def __init__(self, ID):
        self.ID = ID  # ID
        self.__clean_data()  # generate the start point randomly

    # Initial the data
    def __clean_data(self):
        self.path = []  # current ants path
        self.total_distance = 0.0  # current total distance
        self.move_count = 0  # #of moves
        self.current_task = -1  # current stay task
        self.open_table_task = [True for i in Tasks]  # explore the status of the task
        task_index = random.randint(0, len(Tasks) - 1)  # generate the start point randomly
        self.current_task = task_index
        self.path.append(task_index)
        self.open_table_task[task_index] = False
        self.move_count = 1

    # choose next task
    def __choice_next_task(self):
        next_task = -1
        select_tasks_prob = [0.0 for i in Tasks]  # store the probability of choosing next task
        total_prob = 0.0
        # get the probability of choosing next task

        for i in Tasks:
            if self.open_table_task[i]:
                try:
                    # Calculate the probability
                    # 计算概率：与信息素浓度成正比，与距离成反比
                    select_tasks_prob[i] = pow(pheromone_graph[self.current_task][i], ALPHA) * pow(
                        (1.0 / visibility_graph[self.current_task][i]), BETA)
                    total_prob += select_tasks_prob[i]
                except ZeroDivisionError as e:
                    print('Ant ID: {ID}, current task: {current}, target task: {target}'.format(ID=self.ID,
                                                                                                current=self.current_task,
                                                                                                target=i))
                    sys.exit(1)

        # 轮盘选择城市
        if total_prob > 0.0:
            # 产生一个随机概率,0.0-total_prob
            temp_prob = random.uniform(0.0, total_prob)
            for i in Tasks:
                if self.open_table_task[i]:
                    # 轮次相减
                    temp_prob -= select_tasks_prob[i]
                    if temp_prob < 0.0:
                        next_task = i
                        break
        # without selecting the task according to the probability, randomly choose a task
        '''
        if next_task == -1:
             for i in Tasks:
                 if self.open_table_task[i]:
                     next_task = i
                     break
        if (next_task == -1):
            next_task = random.randint(0, len(Tasks) - 1)
            while ((self.open_table_task[next_task]) == False):  # if==False,说明已经遍历过了
                next_task = random.randint(0,len(Tasks) - 1)
        '''
        # return to next task id
        return next_task

    # calculate the total distance(completion time)
    def __cal_total_distance(self):
        temp_distance = 0.0
        for i in range(1, len(self.path)):
            start, end = self.path[i], self.path[i - 1]
            temp_distance += visibility_graph[start][end]
        # 回路
        end = self.path[0]
        temp_distance += visibility_graph[start][end]
        self.total_distance = temp_distance

    # ants movement
    def __move(self, next_task):
        self.path.append(next_task)
        self.open_table_task[next_task] = False
        self.total_distance += visibility_graph[self.current_task][next_task]
        self.current_task = next_task
        self.move_count += 1

    # search for the path
    def search_path(self):
        # initial the data
        # self.__clean_data()
        # search for the path until all the tasks have been assigned
        # while any(self.open_table_task) == True:
        # move to next task
        next_task = self.__choice_next_task()
        # self.__move(next_task)
        # calculate the total distance
        # self.__cal_total_distance()
        return next_task


# -----Scheduling-----
class SCHEDULING(object):
    def __init__(self, n=len(Tasks)):
        self.new()
        # #of tasks is the length of tasks
        self.n = n

        # randomly assign a value to the distance between different tasks
        #for i in Tasks:
            #for j in Tasks:
                #temp_distance = random.randint(225, 450)
                #visibility_graph[i][j] = float(int(temp_distance + 0.5))

        self.search_path()

    def __clean_data(self):
        task_index = -2
        Task_list_Forklift.loc[:, 'check_status'] = True
        Task_list_AGV.loc[:, 'check_status'] = True
        for ant in self.ants:
            ant.path = []
            ant.current_task = -1
            ant.total_distance = 0
            # task_index = random.randint(0, len(Tasks) - 1)  # generate the start point randomly
            task_index += 3
            ant.current_task = task_index
            ant.path.append(task_index)
            if Task_list.at[ant.current_task, 'Task type'] == 'C04_CMD':
                Task_list_Forklift.loc[task_index, 'check_status'] = False
            else:
                Task_list_AGV.loc[task_index, 'check_status'] = False

        # self.path = []               #  current ants path
        self.total_distance = 1.0  # current total distance
        self.move_count = 6  # #of moves
        # self.current_task = -1       # current stay task
        # self.open_table_task = [True for i in Tasks] # explore the status of the task
        # task_index = random.randint(0,len(Tasks)-1) # generate the start point randomly
        # self.current_task = task_index
        # self.path.append(task_index)
        # self.open_table_task[task_index] = False
        # self.move_count = 1

    # Initial
    def new(self):
        # 初始城市之间的距离和信息素
        for i in Tasks:
            for j in Tasks:
                pheromone_graph[i][j] = 1.0
        self.ants = [Ant(ID) for ID in range(Ant_num)]  # initial ants group
        # self.best_ant = Ant(-1)                          # 初始最优解
        # self.best_ant.total_distance = 1 << 31           # 初始最大距离
        self.iter = 1  # initial the iteration number

    # search
    def search_path(self, evt=None):
        # start run
        self.__running = True
        #self.total_completion_time=[]
        self.best_distance = 1000000000
        self.best_iteration = 1
        while self.__running:
            self.__clean_data()
            while self.move_count < len(Tasks):
                # while any(Task_list_Forklift['check_status']) or any(Task_list_AGV['check_status']) == True:
                # visit each ant
                for ant in self.ants:
                    # print(u"迭代次数：", self.iter, ant.path)
                    select_tasks_prob = [0.0 for i in Tasks]  # store the probability of choosing next task
                    total_prob = 0.0
                    if any(Task_list_Forklift['check_status']) or any(Task_list_AGV['check_status']):
                        # search next task
                        # print(ant.current_task)
                        next_task = -1
                        if Task_list.at[ant.current_task, 'Task type'] == 'C04_CMD':
                            for i in Task_list_Forklift['Task ID'].tolist():
                                if Task_list_Forklift.at[i, 'check_status'] == True:
                                    try:
                                        # Calculate the probability
                                        select_tasks_prob[i] = pow(pheromone_graph[ant.current_task][i], ALPHA) * pow(
                                            (1.0 / visibility_graph[ant.current_task][i]), BETA)
                                        total_prob += select_tasks_prob[i]
                                    except ZeroDivisionError as e:
                                        print('Ant ID: {ID}, current task: {current}, target task: {target}'.format(
                                            ID=ant.ID, current=ant.current_task, target=i))
                                        sys.exit(1)
                            # 轮盘堵选择task
                            if total_prob > 0.0:
                                # generate a random probability,0.0-total_prob
                                temp_prob = random.uniform(0.0, total_prob)
                                for i in Task_list_Forklift['Task ID'].tolist():
                                    if Task_list_Forklift.at[i, 'check_status'] == True:
                                        # 轮次相减
                                        temp_prob -= select_tasks_prob[i]
                                        if temp_prob < 0.0:
                                            next_task = i
                                            break
                                    else:
                                        continue
                            if next_task != -1:
                                Task_list_Forklift.loc[next_task, 'check_status'] = False

                        # AGV task scheduling
                        elif Task_list.at[ant.current_task, 'Task type'] != 'C04_CMD' and any(
                                Task_list_AGV['check_status']):
                            # check if predecessors has been visted or not
                            for i in Task_list_AGV['Task ID'].tolist():
                                if Task_list_AGV.at[i, 'check_status'] == True:
                                    try:
                                        # Calculate the probability
                                        select_tasks_prob[i] = pow(pheromone_graph[ant.current_task][i], ALPHA) * pow(
                                            (1.0 / visibility_graph[ant.current_task][i]), BETA)
                                        total_prob += select_tasks_prob[i]
                                    except ZeroDivisionError as e:
                                        print('Ant ID: {ID}, current task: {current}, target task: {target}'.format(
                                            ID=ant.ID, current=ant.current_task, target=i))
                                        sys.exit(1)
                            # 轮盘堵选择task
                            if total_prob > 0.0:
                                # generate a randon probability,0.0-total_prob
                                temp_prob = random.uniform(0.0, total_prob)
                                for i in Task_list_AGV['Task ID'].tolist():
                                    front_task = int(Task_list.at[i, 'Predecessors'])
                                    if front_task == 0:
                                        if Task_list_AGV.at[i, 'check_status']:
                                            # 轮次相减
                                            temp_prob -= select_tasks_prob[i]
                                            if temp_prob < 0.0:
                                                next_task = i
                                                break
                                        else:
                                            continue
                                    elif Task_list_AGV.at[i, 'check_status'] and Task_list_Forklift.at[
                                        front_task, 'check_status'] == False:
                                        # 轮次相减
                                        temp_prob -= select_tasks_prob[i]
                                        if temp_prob < 0.0:
                                            next_task = i
                                            #print(next_task)
                                            break
                                    else:
                                        continue
                            if next_task != -1:
                                Task_list_AGV.loc[next_task, 'check_status'] = False
                        else:
                            continue

                    else:
                        continue
                    # next_task=ant.search_path()
                    if next_task != -1:
                        ant.path.append(next_task)
                        ant.total_distance += visibility_graph[ant.current_task][next_task]
                        ant.current_task = next_task
                        self.move_count += 1
                        # 与当前最优蚂蚁比较
                        # if ant.total_distance < self.best_ant.total_distance:
                        # 更新最优解
                        # self.best_ant = copy.deepcopy(ant)
                        #print(u"the index of iteration：", self.iter, u"Vehicle task list：", ant.path, ant.current_task,
                              #u"Total completion time：", self.total_distance)
                    else:
                        ant.path.append(next_task)
                        #print(u"the index of iteration：", self.iter, u"Vehicle task list：", ant.path, ant.current_task,
                              #u"Total completion time：", self.total_distance)
            self.vehicle_tasklist = {1:{},2:{},3:{},4:{},5:{},6:{}}

            a=1
            for ant in self.ants:
                self.vehicle_tasklist[a]=ant.path
                if self.total_distance <= ant.total_distance:
                    self.total_distance = ant.total_distance

                print('-----the task list of vehicle',a, 'is-----',ant.path,ant.total_distance)
                a+=1


            #record the completion time of each iteration
            #self.total_completion_time.append(self.total_distance)

            #if self.total_distance < self.best_distance:
                #self.best_distance = self.total_distance
                #self.best_iteration=self.iter
            # update the pheromone matrix
            self.__update_pheromone_gragh()
            #print(u"迭代次数：", self.iter, self.total_completion_time)
            self.iter += 1
            if self.iter == 2:
                #print(self.total_completion_time)
                #print(u"the best iteration：", self.best_iteration, u"The optimal total completion time：",self.best_distance)
                #self.__iteration_visulazation()
                return self.vehicle_tasklist, self.total_distance
                break

    # update the pheromone
    def __update_pheromone_gragh(self):
        # obtain the pheromone of each ant at the route
        temp_pheromone = [[0.0 for col in Tasks] for raw in Tasks]
        for ant in self.ants:
            for i in range(1, len(ant.path)):
                start, end = ant.path[i - 1], ant.path[i]
                # 在路径上的每两个相邻城市间留下信息素，与路径总距离反比
                temp_pheromone[start][end] += Q / self.total_distance
                temp_pheromone[end][start] = temp_pheromone[start][end]
        # update all the pheromone，old pheromone decrease with the addition new pheromone
        for i in Tasks:
            for j in Tasks:
                pheromone_graph[i][j] = pheromone_graph[i][j] * RHO + temp_pheromone[i][j] * ALPHA


        # Create a visualization
        '''
        sns.relplot(
            data=dots, kind="line",
            x="time", y="firing_rate", col="align",
            hue="choice", size="coherence", style="choice",
            facet_kws=dict(sharex=False),
        )
        '''
        # ----------- main -----------

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
    for id in range(-1,150):
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

            print('Path found for task',TaskID,':',reconst_path)
            print('The arrive time of each point',arrive_time_lst[TaskID])
            return reconst_path,arrive_time_lst[TaskID]

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


def iteration_visulazation(iter,total_completion_time):
    # Apply the default theme
    sns.set()
    x = range(0, iter - 1)
    y = total_completion_time
    plt.plot(x, y)
    plt.xlabel('The iteration')
    plt.ylabel('The makespan of the operation')
    plt.show()

# ----------- main -----------
if __name__ == '__main__':
    # input road map
    Manufacturing_Graph = pd.read_excel('Graph.xlsx', sheet_name='Sheet2', usecols="A:C", skiprows=0, nrows=27,dtype=object)
    nodes = list(range(0, 25))

    Node1 = Manufacturing_Graph['Node1'].tolist()
    Node2 = Manufacturing_Graph['Node2'].tolist()

    init_graph = {}
    for node in nodes:
        init_graph[node] = {}

    for i in range(0, 27):
        init_graph[Node1[i]][Node2[i]] = Manufacturing_Graph.loc[i,'Distance']

    graph = Graph(nodes, init_graph)

    # main algorithm
    total_completion_time_result = []
    total_travel_time=[]
    iter = 1

    while True:
        print('------------------------iteration', iter,'------------------')
        #scheduling the task
        print('------------------------Task assignment result------------------')
        # Task_assignment = SCHEDULING()
        # #outputfile.close()  # close后才能看到写入的数据
        # Task_assignment_result={}
        # Task_assignment_result=Task_assignment.vehicle_tasklist
        # Total_distance_result=Task_assignment.total_distance
        # total_completion_time_result.append(Total_distance_result)

        Task_assignment_result=[[1, 11, 8],[4, 2, 3],[7, 9, 6],[10, 0, 5],[13, 17, 14],[16, 12, 15]]

        # Task_assignment_result={1: [1, 2, 8, 48, 9, 87, 22, 54, 118, 55, 92, -1, -1, -1, -1, -1, -1, -1, -1, 95, -1, -1, -1, -1, -1, -1, -1, 126, -1, 127],
        #  2: [4, 81, 82, 116, 3, 88, 49, 115, 123, 93, 41, 89, 79, 83, 85, 119, 6, 94, 42, 21, 125, -1, -1, 96, 50, -1, -1, -1, -1],
        #  3: [7, 23, 121, 5, 0, 64, 24, 39, 19, 47, 43, 25, 11, 63, 20, 51, 52, 128, -1, -1, -1, -1, -1, -1, 45, -1, -1, -1, -1],
        #  4: [10, 117, 90, 62, 91, 120, 40, 44, 124, 38, 53, 84, 18, 122, 114, 46, 80, 86, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        #  5: [13, 33, 70, 57, 17, 27, 103, 100, 58, 109, 139, 36, 69, 102, 132, 97, 65, 72, 60, 110, 134, 105, 73, 104, 34, 111, 29, 75, 99, 108, 77, 138, 35],
        #  6: [16, 56, 15, 78, 59, 76, 32, 113, 30, 14, 68, 66, 133, 37, 67, 112, 26, 136, 101, 137, 106, 107, 61, 12, 131, 28, 98, 74, 129, 31, 135, 71, 130]}





        print('---------------------------------route planning---------------------------------------')
        #Route planning dic:arrive time[vehicle id][task id][point:time]
        arrive_time = {}
        arrive_time_back = {}
        for vehicle in range(1,7):
            arrive_time[vehicle] = {}
            arrive_time_back[vehicle] = {}
        for vehicle in range(1,7):
            for id in range(-1,142):
                arrive_time[vehicle][id] = {}
                arrive_time_back[vehicle][id] ={}

        print(Task_list_routing.loc[-1 , 'Start node'])
        #the beginning 6 tasks for each vehicle
        for i in range(1,17,3):
            reconst_path, arrive_time[(i+2)/3][i] = a_star_algorithm(graph, Task_list.loc[i, 'Start node'], Task_list.loc[i, 'End node'],
                                                         speed=Task_list_routing.loc[i, 'speed'], TaskID=i, endtime_lasttask=0+Task_list_routing.loc[i, 'loading time'])
            reconst_path_back, arrive_time_back[(i+2)/3][i]=a_star_algorithm(graph, Task_list.loc[i, 'End node'], Task_list_routing.loc[Task_assignment_result[(i+2)/3][1] , 'Start node'], speed=Task_list_routing.loc[i, 'speed'], TaskID=i,
                             endtime_lasttask=arrive_time[(i+2)/3][i][Task_list.loc[i, 'End node']]+Task_list_routing.loc[i, 'unloading time'])
        #the rest tasks
        for j in range(1,len(Task_assignment_result[1])-2):
            for vehicle in range(1,7):
                if Task_assignment_result[vehicle][j] != -1:
                    print('------------------vehicle------------------',vehicle)
                    reconst_path, arrive_time[vehicle][Task_assignment_result[vehicle][j]] = a_star_algorithm(graph, Task_list_routing.loc[Task_assignment_result[vehicle][j], 'Start node'],
                                                                 Task_list_routing.loc[Task_assignment_result[vehicle][j], 'End node'], speed=Task_list_routing.loc[Task_assignment_result[vehicle][j], 'speed'], TaskID=Task_assignment_result[vehicle][j],
                                                                 endtime_lasttask=arrive_time_back[vehicle][Task_assignment_result[vehicle][j-1]][Task_list_routing.loc[Task_assignment_result[vehicle][j], 'Start node']]+Task_list_routing.loc[Task_assignment_result[vehicle][j], 'loading time'])

                    reconst_path_back, arrive_time_back[vehicle][Task_assignment_result[vehicle][j]] = a_star_algorithm(graph, Task_list_routing.loc[Task_assignment_result[vehicle][j], 'End node'],
                                                                  Task_list_routing.loc[Task_assignment_result[vehicle][j+1] , 'Start node'], speed=Task_list_routing.loc[Task_assignment_result[vehicle][j], 'speed'], TaskID=Task_assignment_result[vehicle][j],
                                                                  endtime_lasttask=arrive_time[vehicle][Task_assignment_result[vehicle][j]][Task_list_routing.loc[Task_assignment_result[vehicle][j], 'End node']]+Task_list_routing.loc[Task_assignment_result[vehicle][j], 'unloading time'])
                else:
                    print('------------------vehicle------------------',vehicle)
                    idle_time=random.randint(180, 360)
                    reconst_path, arrive_time[vehicle][Task_assignment_result[vehicle][j]] = a_star_algorithm(graph, Task_list_routing.loc[Task_assignment_result[vehicle][j], 'Start node'],
                                                                 Task_list_routing.loc[Task_assignment_result[vehicle][j], 'End node'], speed=Task_list_routing.loc[Task_assignment_result[vehicle][j], 'speed'], TaskID=Task_assignment_result[vehicle][j],
                                                                 endtime_lasttask=arrive_time_back[vehicle][Task_assignment_result[vehicle][j-1]][Task_list_routing.loc[Task_assignment_result[vehicle][j], 'Start node']] + idle_time)

                    reconst_path_back, arrive_time_back[vehicle][Task_assignment_result[vehicle][j]] = a_star_algorithm(graph, Task_list_routing.loc[Task_assignment_result[vehicle][j], 'End node'],
                                                                  Task_list_routing.loc[Task_assignment_result[vehicle][j+1] , 'Start node'], speed=Task_list_routing.loc[Task_assignment_result[vehicle][j], 'speed'], TaskID=Task_assignment_result[vehicle][j],
                                                                  endtime_lasttask=arrive_time[vehicle][Task_assignment_result[vehicle][j]][Task_list_routing.loc[Task_assignment_result[vehicle][j], 'End node']])
        iter +=1

        # find the maximum value in the arrive time of all vehicles
        arrive_time_back_max = max(arrive_time_back[1][Task_assignment_result[1][len(Task_assignment_result[1])-3]].values())
        for vehicle in range(2,7):
            if max(arrive_time_back[vehicle][Task_assignment_result[vehicle][len(Task_assignment_result[1])-3]].values()) >= arrive_time_back_max:
                arrive_time_back_max = max(arrive_time_back[vehicle][Task_assignment_result[vehicle][len(Task_assignment_result[1])-3]].values())

        total_travel_time.append(arrive_time_back_max)
        if iter == 50:
            #result of scheduling without considering the idle time,routing
            iteration_visulazation(iter, total_completion_time_result)
            #result after routing
            iteration_visulazation(iter,total_travel_time)
            break
