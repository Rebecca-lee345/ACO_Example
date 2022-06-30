# -*- coding: utf-8 -*-
import random
import copy
import time
import sys
import math
import tkinter #//GUI模块
import threading
from functools import reduce
import numpy as np
import pandas as pd

# 参数
'''
ALPHA:信息启发因子，值越大，则蚂蚁选择之前走过的路径可能性就越大
      ，值越小，则蚁群搜索范围就会减少，容易陷入局部最优
BETA:Beta值越大，蚁群越就容易选择局部较短路径，这时算法收敛速度会
     加快，但是随机性不高，容易得到局部的相对最优
'''
(ALPHA, BETA, RHO, Q) = (1.0,2.0,0.5,100.0)
# ----Ant----
Ant_num= 6

#Input task list
Task_list = pd.read_excel('Task list_Test.xlsx', sheet_name='Tasklist', usecols="A:U",skiprows=0, nrows=10)

#Parameters definition
Line = list(range(0,3)) #the number of lines in the roadmap
Stop = list(range(0,450)) #the number of stops in each line
Forklifts = list(range(0,2)) #the number of forklifts
AGVs = list(range(0,4)) #the number of AGVs
Tasks = Task_list['Task ID'].tolist() #the task ID
H = list(range(0,600)) # the avaulable time (s), for now 1 hour



#pheromone matrix & Ant visibility matrix
pheromone_graph = [ [1.0 for col in Tasks] for raw in Tasks]
visibility_graph = [ [0.0 for col in Tasks] for raw in Tasks]

#-----Ants-----
class Ant(object):
    # 初始化
    def __init__(self,ID):
        self.ID = ID                 # ID
        self.__clean_data()          # generate the start point randomly 随机初始化出生点
    # 初始数据
    def __clean_data(self):
        self.path = []               #  current ants path当前蚂蚁的路径
        self.total_distance = 0.0    # 当前路径的总距离
        self.move_count = 0          # # of moves 移动次数
        self.current_task = -1       # current stay task 当前停留的城市
        self.open_table_task = [True for i in Tasks] # explore the status of the task 探索城市的状态
        task_index = random.randint(0,len(Tasks)-1) # generate the start point randomly 随机初始出生点
        self.current_task = task_index
        self.path.append(task_index)
        self.open_table_task[task_index] = False
        self.move_count = 1
    # choose next task 选择下一个城市
    def __choice_next_task(self):
        next_task = -1
        select_tasks_prob = [0.0 for i in Tasks]  # store the probabilty of choosing next task 存储去下个城市的概率
        total_prob = 0.0
        # get the probability of choosing next task获取去下一个城市的概率
        for i in Tasks:
            if self.open_table_task[i]:
                try :
                    # 计算概率：与信息素浓度成正比，与距离成反比
                    select_tasks_prob[i] = pow(pheromone_graph[self.current_task][i], ALPHA) * pow((1.0/visibility_graph[self.current_task][i]), BETA)
                    total_prob += select_tasks_prob[i]
                except ZeroDivisionError as e:
                    print ('Ant ID: {ID}, current task: {current}, target task: {target}'.format(ID = self.ID, current = self.current_task, target = i))
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
        # 未从概率产生，顺序选择一个未访问城市
        # if next_city == -1:
        #     for i in range(city_num):
        #         if self.open_table_city[i]:
        #             next_city = i
        #             break
        if (next_task == -1):
            next_task = random.randint(0, len(Tasks) - 1)
            while ((self.open_table_task[next_task]) == False):  # if==False,说明已经遍历过了
                next_task = random.randint(0,len(Tasks) - 1)
        # 返回下一个城市序号
        return next_task
    # 计算路径总距离
    def __cal_total_distance(self):
        temp_distance = 0.0
        for i in Tasks:
            start, end = self.path[i], self.path[i-1]
            temp_distance += visibility_graph[start][end]
        # 回路
        end = self.path[0]
        temp_distance += visibility_graph[start][end]
        self.total_distance = temp_distance

    # 移动操作
    def __move(self, next_task):
        self.path.append(next_task)
        self.open_table_task[next_task] = False
        self.total_distance += visibility_graph[self.current_task][next_task]
        self.current_task = next_task
        self.move_count += 1
    # 搜索路径
    def search_path(self):
        # 初始化数据
        self.__clean_data()
        # 搜素路径，遍历完所有城市为止
        while self.move_count < len(Tasks):
            # 移动到下一个城市
            next_task =  self.__choice_next_task()
            self.__move(next_task)
        # 计算路径总长度
        self.__cal_total_distance()