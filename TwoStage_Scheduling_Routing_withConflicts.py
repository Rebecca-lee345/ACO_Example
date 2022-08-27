# two stage ACO scheduling and routing with collision avoidance


# -*- coding: utf-8 -*-
import random
import copy
import time
import sys
import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pickle



# ---------------------------------------parameters-------------------------------------
(ALPHA, BETA, LAMDA, RHO, Q) = (1.0,2.0,3.0,0.5,100.0)


#-------------------------------------------Roadmap input----------------------------
Manufacturing_Graph = pd.read_excel('Graph_25.xlsx', sheet_name='Sheet2', usecols="A:C", skiprows=0, nrows=27,dtype=object)
#nodes = list(range(0, 41))
nodes = list(range(0, 25))


Node1 = Manufacturing_Graph['Node1'].tolist()
Node2 = Manufacturing_Graph['Node2'].tolist()

init_graph = {}
for node in nodes:
    init_graph[node] = {}

# for i in range(0, 43):
#     init_graph[Node1[i]][Node2[i]] = Manufacturing_Graph.loc[i,'Distance']

for i in range(0, 27):
    init_graph[Node1[i]][Node2[i]] = Manufacturing_Graph.loc[i,'Distance']


#------------------------------------------graph: roadmap for routing -------------------
class Graph(object):
    def __init__(self, nodes, init_graph):
        self.nodes = nodes
        self.graph = self.construct_graph(nodes, init_graph)

    def construct_graph(self, nodes, init_graph):

        #This method makes sure that the graph is symmetrical. In other words, if there's a path from node A to B with a value V, there needs to be a path from node B to node A with a value V.

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

#------------------------------------------------------ Ant for scheduling ----------------------------
class SAnt(object):
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

#-------------------------------------------------------Ant colony Scheduling-------------------------------------------------
# ----Ant----
Ant_num = 6

# Input task list
Task_list = pd.read_excel('Task_list.xlsx', sheet_name='Tasklist', usecols=[0,3,6,7,8,9,10,11], skiprows=0, nrows=141, dtype=object)
Task_list_Forklift = Task_list.loc[Task_list['Task type'] == 'C04_CMD']
Task_list_AGV = Task_list.loc[Task_list['Task type'] != 'C04_CMD']
# input visibility graph for scheduling
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
        self.ants = [SAnt(ID) for ID in range(Ant_num)]  # initial ants group
        # self.best_ant = SAnt(-1)                          # 初始最优解
        # self.best_ant.total_distance = 1 << 31           # 初始最大距离
        self.iter = 1  # initial the iteration number

    # search
    def search_path(self, evt=None):
        # start run
        self.__running = True
        self.total_completion_time=[]
        self.best_distance = 1000000000
        self.best_iteration = 1
        self.best_vehicle_tasklist = {}
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


            self.vehicle_tasklist = {0:{},1:{},2:{},3:{},4:{},5:{}}
            a = 0
            for ant in self.ants:
                self.vehicle_tasklist[a] = ant.path
                if self.total_distance <= ant.total_distance:
                    self.total_distance = ant.total_distance

                print('-----the task list of vehicle', a, 'is-----', ant.path, ant.total_distance)
                a += 1

            if self.total_distance < self.best_distance:
                self.best_distance = self.total_distance
                self.best_iteration=self.iter
                self.best_vehicle_tasklist=self.vehicle_tasklist



            #record the completion time of each iteration
            self.total_completion_time.append(self.total_distance)


            # update the pheromone matrix
            self.__update_pheromone_gragh()
            #print(u"迭代次数：", self.iter, self.total_completion_time)
            self.iter += 1
            if self.iter == 100:
                #print(self.total_completion_time)
                print(u"the best iteration：", self.best_iteration, u"The optimal total completion time：",self.best_distance,u"The optimal scheduling result：",self.best_vehicle_tasklist)
                self.__iteration_visulazation()
                return self.best_vehicle_tasklist, self.total_distance
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

    def __iteration_visulazation(self):
        # Apply the default theme
        sns.set()
        x=range(0,self.iter-1)
        y=self.total_completion_time
        plt.plot(x, y)
        plt.xlabel('The iteration')
        plt.ylabel('The makespan of the operation')
        plt.show()

#scheduling the task
print('------------------------Task assignment result------------------')
#Task_assignment = SCHEDULING()

#scheduling_result={}
#scheduling_result=Task_assignment.vehicle_tasklist

scheduling_result=[[1, 11, 8],[4, 2, 3],[7, 9, 6],[10, 0, 5],[13, 17, 14],[16, 12, 15]]

# scheduling_result=[[1, 2, 8, 48, 9, 87, 22, 54, 118, 55, 92, -1, -1, -1, -1, -1, -1, -1, -1, 95, -1, -1, -1, -1, -1, -1, -1, 126, -1, 127],
#     [4, 81, 82, 116, 3, 88, 49, 115, 123, 93, 41, 89, 79, 83, 85, 119, 6, 94, 42, 21, 125, -1, -1, 96, 50, -1, -1, -1, -1],
#  [7, 23, 121, 5, 0, 64, 24, 39, 19, 47, 43, 25, 11, 63, 20, 51, 52, 128, -1, -1, -1, -1, -1, -1, 45, -1, -1, -1, -1],
#  [10, 117, 90, 62, 91, 120, 40, 44, 124, 38, 53, 84, 18, 122, 114, 46, 80, 86, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
#  [13, 33, 70, 57, 17, 27, 103, 100, 58, 109, 139, 36, 69, 102, 132, 97, 65, 72, 60, 110, 134, 105, 73, 104, 34, 111, 29, 75, 99, 108, 77, 138, 35],
#  [16, 56, 15, 78, 59, 76, 32, 113, 30, 14, 68, 66, 133, 37, 67, 112, 26, 136, 101, 137, 106, 107, 61, 12, 131, 28, 98, 74, 129, 31, 135, 71, 130]]



#----------------------------------------------------------------------- Ant colony routing -----------

graph_input = Graph(nodes, init_graph)

# Open pickle file and load it in a dictionary
pickle_file = open("VisibilityGraphACO_41.pkl", "rb")
visibility_graph_ACO = pickle.load(pickle_file)

#Task list input
Task_list_routing = pd.read_excel('Task_list_ACO.xlsx', sheet_name='Tasklist_routing', index_col=0,usecols=[0,3,6,7,8,9,10,11], skiprows=0, nrows=142,
                              dtype=object)

#----------------------------Routing parameters------------------------------------
#safety distance between two vehicles on the link set as 4 seconds
xi = 4
# safe time difference between two vehicles at the same point (seconds)
Ht = 0.2
#the maximum number of conflicts density on link
Ha= 2
#the maximun number of conflicts at node
Hb = 2

#penalty coefficient gamma 20s
gamma = 20
#maximum iteration number
Max_iteration =100

#(node_num, ant_num_routing) = (41,10)
(node_num, ant_num_routing) = (25,10)
#visibility graph---the distance between nodes  and pheromone graph
#visibility_graph = [ [0.0 for col in range(node_num)] for raw in range(node_num)]
pheromone_graph_routing = [ [1.0 for col in range(node_num)] for raw in range(node_num)]
#store the collision factor
collision_graph = [ [1.0 for col in range(node_num)] for raw in range(node_num)]

#Particle matrix


#---------------------------------------------------Ant for routing---------------------------------------
class RAnt(object):
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
        # self.antNY = [ [0.0 for col in range(node_num)] for raw in range(node_num)]
        # self.antNZ = [ 0.0 for col in range(node_num)]
        # self.list_current=[0,0]
        # self.list_other =[0,0]
        # self.current_nodetime = 0
        # self.real_total_distance = 0


class ACO_Routing(object):
    def __init__(self, n = node_num,graph=graph_input, scheudling_list=scheduling_result):

        # initial node number is n
        self.n = n
        #self.new()
        # calculate the distance between nodes
        # for i in range(node_num):
        #     for j in range(node_num):
        #         temp_distance = random.randint(225, 450)
        #         visibility_graph[i][j] = float(int(temp_distance + 0.5))
        #search the path
        self.search_path(graph,scheudling_list)


    # 初始化
    def new(self):
        #self.__running = False
        #this list is used in to link collision check
        self.list_current=[0,0]
        self.list_other =[0,0]
        current_nodetime = 0
        # 初始城市之间的距离和信息素
        for i in range(node_num):
            for j in range(node_num):
                pheromone_graph_routing[i][j] = 1.0
        self.ants = [RAnt(ID) for ID in range(ant_num_routing)]  # initial ant number
        self.best_ant = RAnt(-1)                          # initial best ant
        self.best_ant.total_distance = 1 << 31           # initial maximum distance for each ant
        for ant in self.ants:
            ant.path = []
            ant.total_distance = 0
            #ant.real_total_distance = 0

    def __clean_data(self):
        # create a dic to store the best path for each vehicle each task
        vehicle_best_path = {}
        for vehicle in range(6):
            vehicle_best_path[vehicle] = {}
        for vehicle in range(6):
            for task_id in range(-1, 150):
                vehicle_best_path[vehicle][task_id] = {}

    # Function to check if there are collision on each link
    def check_link_collision(self,check_list):
        """校验是否重叠"""

        if len(check_list) < 1:
            return False
        if len(check_list) == 1:
            return True
        #print('check_list : ', check_list)
        check_list[0] = list(map(int, check_list[0]))
        check_list[1] = list(map(int, check_list[1]))

        for i in range(0, len(check_list)):
            if check_list[i][0]> check_list[i][1]:
                check_list[i][0], check_list[i][-1] = check_list[i][-1], check_list[i][0]
        # 排序
        a_list = sorted(check_list, key=lambda l: l[0])
        #print('a_list : ', a_list)
        for i in range(0, len(a_list) - 1):
            if a_list[i + 1][0] <= a_list[i][1]:
                #print('%s 与 %s 重叠!' % (str(a_list[i]), str(a_list[i + 1])))
                return False



    #Fitness function of single ant
    def vehicle_fitness(self,NY,NZ,vehicle_total_distance):
        temp_pho= [[0.0 for col in nodes] for raw in nodes]
        sum_link=0
        sum_node=0
        for i in nodes:
            for j in nodes:
                temp_pho[i][j] = (NY[i][j]+1)
                sum_link += max(0, temp_pho[i][j]-Ha)
        for i in nodes:
            sum_node += max(0, NZ[i]+1-Hb)
        fitness_result= vehicle_total_distance + gamma * (sum_link + sum_node)
        return fitness_result


    def search_path(self, graph, scheudling_list):
        # 开启线程
        self.__running = True
        self.iter = 1
        self.iter_total_completion_time=[]

        self.NY = [[0.0 for col in range(node_num)] for raw in range(node_num)]
        self.NZ = [0.0 for col in range(node_num)]

        # create the list to store the each vehicle fitness function
        self.vehicle_fitness_result = [[] for raw in range(6)]
        #create the list to record the total total fitness per iteration:Fitness calculation of historical optimal AGV group.minf max fpmeanfp
        #self.fitness_result=[]


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

            #create the dictionary to store the task finish time at each vehicle,at each point
            task_completion_time_forward={}
            task_completion_time_back = {}
            for vehicle in range(6):
                task_completion_time_forward[vehicle]={}
                task_completion_time_back[vehicle] = {}
            for vehicle in range(6):
                for task_id in range(-1,150):
                    task_completion_time_forward[vehicle][task_id] = {}
                    task_completion_time_back[vehicle][task_id] = {}
            for vehicle in range(6):
                for task_id in range(-1,150):
                    for point in range(0,node_num):
                        task_completion_time_forward[vehicle][task_id][point] = 0
                        task_completion_time_back[vehicle][task_id][point] = 0




            #create the 3-d list to record the link collision for each vehicle
            a = [[0.0 for col in nodes] for raw in nodes]
            self.vehicleNY = [a for i in range(6)]
            # create the 2-d list to record the link collision for each vehicle
            self.vehicleNZ = [[0.0 for col in range(node_num)] for raw in range(6)]








            print('--------------iter',self.iter)
            for task_location in range(len(scheudling_list[0])-2):
                for vehicle in range(6):
                    if scheudling_list[vehicle][task_location] != -1:
                        self.new()
                        # go through each ant for the forward path searching
                        for ant in self.ants:
                            # search a path
                            ant.current_node=Task_list_routing.loc[scheudling_list[vehicle][task_location],'Start node']
                            ant.path.append(ant.current_node)
                            target_node= Task_list_routing.loc[scheudling_list[vehicle][task_location],'End node']

                            while ant.current_node != target_node:
                                next_node = -1
                                # print(u"迭代次数：", self.iter, ant.path)
                                neighbors = graph.get_outgoing_edges(ant.current_node)
                                #neighbors.append(ant.current_node)
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
                                        select_nodes_prob[i] = pow(pheromone_graph_routing[ant.current_node][i],ALPHA) * pow((1.0 / visibility_graph_ACO[scheudling_list[vehicle][task_location]][ant.current_node][i]), BETA) * pow(collision_graph[ant.current_node][i],LAMDA)
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

                        # update the pheromone matrix
                        self.__update_pheromone_graph()
                        #print (u"vehicle number：",vehicle,u"最佳路径总距离：",int(self.best_ant.total_distance))
                        #print(u"the index of iteration：", self.iter, self.best_ant.path, ant.current_node)
                        # store the best path for each vehicle each task for the forward path searching
                        vehicle_best_path[vehicle][scheudling_list[vehicle][task_location]] =self.best_ant.path
                        #vehicle的第一个点的开始时间等于vehicle完成上一个任务的完成时间
                        task_completion_time_forward[vehicle][scheudling_list[vehicle][task_location]][self.best_ant.path[0]]=total_completion_time[vehicle]
                        # each vehicle's total completion time
                        total_completion_time[vehicle] += self.best_ant.total_distance/Task_list_routing.loc[scheudling_list[vehicle][task_location],'speed']+Task_list_routing.loc[scheudling_list[vehicle][task_location],'loading time']

                        #vehicle 在每一个点的到达时间
                        for point in range(1,len(self.best_ant.path)):
                            task_completion_time_forward[vehicle][scheudling_list[vehicle][task_location]][self.best_ant.path[point]] = init_graph[self.best_ant.path[point-1]][self.best_ant.path[point]] / Task_list_routing.loc[scheudling_list[vehicle][task_location],'speed']+ task_completion_time_forward[vehicle][scheudling_list[vehicle][task_location]][self.best_ant.path[point-1]]

                        #check if there is a collision on the link
                        for point in range(0, len(self.best_ant.path)-1):
                            self.list_current[0:2]=[task_completion_time_forward[vehicle][scheudling_list[vehicle][task_location]][self.best_ant.path[point]] - xi,task_completion_time_forward[vehicle][scheudling_list[vehicle][task_location]][self.best_ant.path[point+1]] + xi]
                            for i in range(0,6):
                                for t in range(0,150):
                                    if task_completion_time_forward[i][t][self.best_ant.path[point]] != 0 and task_completion_time_forward[i][t][self.best_ant.path[point+1]] != 0:
                                        self.list_other[0:2] = [task_completion_time_forward[i][t][self.best_ant.path[point]] ,task_completion_time_forward[i][t][self.best_ant.path[point+1]]]

                                        check_list = [self.list_current] + [self.list_other]
                                        result_collision = self.check_link_collision(check_list)
                                        if result_collision == False:
                                            self.NY[self.best_ant.path[point]][self.best_ant.path[point+1]] +=1
                                            self.NY[self.best_ant.path[point+1]][self.best_ant.path[point]] += 1
                                            self.vehicleNY[vehicle][self.best_ant.path[point]][self.best_ant.path[point+1]] +=1
                                            self.vehicleNY[vehicle][self.best_ant.path[point + 1]][self.best_ant.path[point]] += 1



                        # check if there is a collision on the Node
                        for point in range(0, len(self.best_ant.path)):
                            current_nodetime = task_completion_time_forward[vehicle][scheudling_list[vehicle][task_location]][self.best_ant.path[point]]
                            for i in range(0,6):
                                for t in range(0,150):
                                    if task_completion_time_forward[i][t][self.best_ant.path[point]] != 0:
                                        time_difference = abs(current_nodetime-task_completion_time_forward[i][t][self.best_ant.path[point]])
                                        if time_difference < Ht:
                                            self.NZ[self.best_ant.path[point]] +=1
                                            self.vehicleNZ[vehicle][self.best_ant.path[point]] +=1

                        #update the collision avoidance factor
                        #print(self.NZ)
                        self.__update_collision_graph()










                        #-------------------------------------------------------------------------
                        #backward path searching for each vehicle each task
                        self.new()
                        # go through each ant for the back path searching
                        for ant in self.ants:
                            # search a path
                            ant.current_node = Task_list_routing.loc[scheudling_list[vehicle][task_location], 'End node']
                            ant.path.append(ant.current_node)
                            target_node = Task_list_routing.loc[scheudling_list[vehicle][task_location+1], 'Start node']

                            while ant.current_node != target_node:
                                next_node = -1
                                # print(u"迭代次数：", self.iter, ant.path)
                                neighbors = graph.get_outgoing_edges(ant.current_node)
                                #neighbors.append(ant.current_node)
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
                                        select_nodes_prob[i] = pow(pheromone_graph_routing[ant.current_node][i], ALPHA) * pow((1.0 / visibility_graph_ACO[scheudling_list[vehicle][task_location]][ant.current_node][i]), BETA) * pow(collision_graph[ant.current_node][i],LAMDA)
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
                        self.__update_pheromone_graph()
                        #print(u"vehicle number：", vehicle, u"最佳路径总距离：", int(self.best_ant.total_distance))
                        #print(u"the index of iteration：", self.iter, self.best_ant.path, ant.current_node)
                        # store the best path for each vehicle each task for the forward path searching
                        vehicle_best_path[vehicle][scheudling_list[vehicle][task_location]].extend(self.best_ant.path)

                        #vehicle的第一个点的开始时间等于vehicle完成上一个任务的完成时间
                        task_completion_time_back[vehicle][scheudling_list[vehicle][task_location]][self.best_ant.path[0]]=total_completion_time[vehicle]

                        total_completion_time[vehicle] += self.best_ant.total_distance/Task_list_routing.loc[scheudling_list[vehicle][task_location],'speed']+Task_list_routing.loc[scheudling_list[vehicle][task_location],'unloading time']

                        for point in range(1,len(self.best_ant.path)):
                            task_completion_time_back[vehicle][scheudling_list[vehicle][task_location]][self.best_ant.path[point]] = init_graph[self.best_ant.path[point-1]][self.best_ant.path[point]] / Task_list_routing.loc[scheudling_list[vehicle][task_location],'speed']+ task_completion_time_back[vehicle][scheudling_list[vehicle][task_location]][self.best_ant.path[point-1]]

                        #check link collision
                        for point in range(0, len(self.best_ant.path)-1):
                            self.list_current[0:2]=[task_completion_time_forward[vehicle][scheudling_list[vehicle][task_location]][self.best_ant.path[point]] - xi,task_completion_time_forward[vehicle][scheudling_list[vehicle][task_location]][self.best_ant.path[point+1]] + xi]

                            for i in range(0,6):
                                for t in range(0,150):
                                    if task_completion_time_forward[i][t][self.best_ant.path[point]] != 0 and task_completion_time_forward[i][t][self.best_ant.path[point+1]] != 0:
                                        self.list_other[0:2] = [task_completion_time_forward[i][t][self.best_ant.path[point]] ,task_completion_time_forward[i][t][self.best_ant.path[point+1]]]
                                        check_list = [self.list_current] + [self.list_other]
                                        result_collision = self.check_link_collision(check_list)
                                        if result_collision == False:
                                            self.NY[self.best_ant.path[point]][self.best_ant.path[point + 1]] += 1
                                            self.NY[self.best_ant.path[point + 1]][self.best_ant.path[point]] += 1
                                            self.vehicleNY[vehicle][self.best_ant.path[point]][self.best_ant.path[point + 1]] += 1
                                            self.vehicleNY[vehicle][self.best_ant.path[point + 1]][self.best_ant.path[point]] += 1

                        # check Node collision
                        for point in range(0, len(self.best_ant.path)):
                            current_nodetime = task_completion_time_forward[vehicle][scheudling_list[vehicle][task_location]][self.best_ant.path[point]]
                            for i in range(0,6):
                                for t in range(0,150):
                                    if task_completion_time_forward[i][t][self.best_ant.path[point]] != 0:
                                        time_difference = abs(current_nodetime-task_completion_time_forward[i][t][self.best_ant.path[point]])
                                        if time_difference < Ht:
                                            self.NZ[self.best_ant.path[point]] +=1
                                            self.vehicleNZ[vehicle][self.best_ant.path[point]] += 1

                        # update the collision avoidance factor
                        #print('back ward collision')
                        self.__update_collision_graph()



                    else:
                        idle_time = random.randint(180, 240)
                        total_completion_time[vehicle]+=idle_time

            #calculate the vehicle fitness function at each iteration
            for i in range(0,6):
                self.vehicle_fitness_result[i].append(self.vehicle_fitness(self.vehicleNY[i],self.vehicleNZ[i],total_completion_time[vehicle]))

            print('Vehicle path',vehicle_best_path)
            print(total_completion_time)
            print('Vehicle completion_time_forward',task_completion_time_forward)
            print('------------------------------------------way back---------------------------------------------------------')
            print('Vehicle completion_time_forward',task_completion_time_back)

            #record each iteration maximum completion time
            self.iter_total_completion_time.append(max(total_completion_time))

            self.iter += 1
            if self.iter == Max_iteration:
                print(vehicle_best_path)
                print(total_completion_time)
                print(self.iter_total_completion_time)

                self.vehicle_fitness_result = np.array(self.vehicle_fitness_result)
                self.fitness_result=np.max(self.vehicle_fitness_result, axis=0)+ np.mean(self.vehicle_fitness_result, axis=0)
                print(self.fitness_result)


                print(task_completion_time_forward)
                print('------------------------------------------way back---------------------------------------------------------')
                print(task_completion_time_back)
                return self.iter_total_completion_time,self.NY,self.NZ,self.fitness_result

                self.__running= False
            else:
                continue


    # update pheromone
    def __update_pheromone_graph(self):
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
                pheromone_graph_routing[i][j] = pheromone_graph_routing[i][j] * RHO + temp_pheromone[i][j] * ALPHA

    #update the collision avodiance factor
    def __update_collision_graph(self):
    #update the collision via the #of link conflicts and #of node conflicts
        for i in nodes:
            for j in nodes:
                collision_graph[i][j] = 1.0 / ((self.NY[i][j]+1) * (self.NZ[i]+1))





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
    iteration_visulazation(Max_iteration, total_completion_time_result)

    NumberofLinkconflicts=Routing.NY
    NumberofNodeconflicts = Routing.NZ


