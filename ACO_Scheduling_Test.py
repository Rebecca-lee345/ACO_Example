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
Task_list = pd.read_excel('Task_list.xlsx', sheet_name='Tasklist', usecols="A:G", skiprows=0, nrows=141, dtype=object)
Task_list_Forklift = Task_list.loc[Task_list['Task type'] == 'C04_CMD']
Task_list_AGV = Task_list.loc[Task_list['Task type'] != 'C04_CMD']

# Task_list_Forklift = Task_list_Forklift.astype({'Task ID': 'int32'}).dtypes

# Parameters definition
Line = list(range(0, 3))  # the number of lines in the roadmap
Stop = list(range(0, 450))  # the number of stops in each line
Forklifts = list(range(0, 2))  # the number of forklifts
AGVs = list(range(0, 4))  # the number of AGVs
Tasks = Task_list['Task ID'].tolist()  # the task ID
H = list(range(0, 600))  # the avaulable time (s), for now 1 hour

# pheromone matrix & Ant visibility matrix
pheromone_graph = [[1.0 for col in Tasks] for raw in Tasks]
visibility_graph = [[0.0 for col in Tasks] for raw in Tasks]


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
        for i in Tasks:
            for j in Tasks:
                temp_distance = random.randint(1, 450)
                visibility_graph[i][j] = float(int(temp_distance + 0.5))

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
        self.total_completion_time=[]
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
                                            print(next_task)
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
                        print(u"the index of iteration：", self.iter, u"Vehicle task list：", ant.path, ant.current_task,
                              u"Total completion time：", self.total_distance)
                    else:
                        ant.path.append(next_task)
                        print(u"the index of iteration：", self.iter, u"Vehicle task list：", ant.path, ant.current_task,
                              u"Total completion time：", self.total_distance)
            vehicle_tasklist = {1:{},2:{},3:{},4:{},5:{},6:{}}

            a=1
            for ant in self.ants:
                vehicle_tasklist[a]=ant.path
                if self.total_distance <= ant.total_distance:
                    self.total_distance = ant.total_distance
                print('-----the task list of vehicle',a, 'is-----',ant.path,ant.total_distance)
                a+=1


            #record the completion time of each iteration
            self.total_completion_time.append(self.total_distance)

            if self.total_distance < self.best_distance:
                self.best_distance = self.total_distance
                self.best_iteration=self.iter
            # update the pheromone matrix
            self.__update_pheromone_gragh()
            #print(u"迭代次数：", self.iter, self.total_completion_time)
            self.iter += 1
            if self.iter == 100:
                print(self.total_completion_time)
                print(u"the best iteration：", self.best_iteration, u"The optimal total completion time：",self.best_distance)
                self.__iteration_visulazation()
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
if __name__ == '__main__':
    SCHEDULING()