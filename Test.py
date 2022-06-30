#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 14:29:57 2020

@author: lichen
"""

import pandas as pd
import numpy as np
from gurobipy import *

file_name = 'data_assignmentQ2.xlsx'
data = pd.read_excel(file_name, sheet_name='data_a', usecols="A:I", nrows=14, engine='openpyxl')
data.loc[13] = [13, 40, 50, 0, 0, 1236, 0, 0, 0]
Node = data['LOC_ID'].tolist()  # Node Number
X = data['XCOORD'].tolist()  # X Coordinate
Y = data['YCOORD'].tolist()  # Y Coordinate
q = data['DEMAND'].tolist()  # demand
a = data['READYTIME'].tolist()  # Earlist Service Start
b = data['DUETIME'].tolist()  # Latest Service Start
s = data['SERVICETIME'].tolist()  # servive time
pick_id = data['PICKUP_ID'].tolist()
deliver_id = data['DELIVER_ID'].tolist()

N = range(len(Node))
visit_node = []
for node in Node:
    if q[node] > 0:
        visit_node.append(node)

visit_node2 = []
for node in Node:
    if q[node] >= 0:
        visit_node2.append(node)

# c: Distance matrix between each node
c = np.zeros((len(Node), len(Node)))
for i in Node:
    row = [np.hypot(X[i] - X[j], Y[i] - Y[j]) for j in Node]
    c[i] = row

model = Model('TSP')
# define variables
x = {}  # whether truck travels from i to j
for i in Node:
    for j in Node:
        x[i, j] = model.addVar(vtype=GRB.BINARY, name="x[" + str(i) + "," + str(j) + "]")

w = {}  # the start time of service at i
for i in N:
    w[i] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="w[" + str(i) + "]")

model.update()

# objective function
model.setObjective(quicksum(c[i][j] * x[i, j] for i in visit_node2 for j in visit_node2))
model.modelSense = GRB.MINIMIZE
model.update()

# constraints
for i in Node:
    model.addConstr(x[i, i] == 0)

for i in visit_node:
    model.addConstr(quicksum(x[i, j] for j in visit_node2) == 1)  # C1 every i has a destination j

for j in visit_node:
    model.addConstr(quicksum(x[i, j] for i in visit_node2) == 1)  # C2	every j has a origin i

for i in Node:
    for j in Node:
        model.addConstr(w[i] + s[i] + c[i][j] - 1000000 * (1 - x[i, j]) <= w[j])  # c3 	service time constrain

for i in Node:
    model.addConstr(w[i] <= b[i])
    model.addConstr(a[i] <= w[i])
'''
model.addConstr(a[0] <= w[0])
model.addConstr(w[13] <= b[0])

'''
model.addConstr(quicksum(x[0, j] for j in visit_node) == 1)
model.addConstr(quicksum(x[i, 13] for i in visit_node) == 1)

model.update()
model.setParam('OutputFlag', True)
# model.setParam ('MIPGap', 0);
model.optimize()
# model.computeIIS()
# model.write('model_problem.ilp')
status = model.status
# model.write("output.lp")

if status == GRB.Status.UNBOUNDED:
    print('The model cannot be solved because it is unbounded')

elif status == GRB.Status.OPTIMAL or True:
    f_objective = model.objVal
    print('***** RESULTS ******')
    print('\nObjective Function Value: \t %g' % f_objective)

elif status != GRB.Status.INF_OR_UNBD and status != GRB.Status.INFEASIBLE:
    print('Optimization was stopped with status %d' % status)

load_immediate = 0
load = []
route = [0]
for i in Node:
    for j in Node:
        if x[i, j].X >= 1:
            print(i, j, x[i, j].X, w[j].X)

for i in route:
    for j in Node:
        if x[i, j].X >= 1:
            load_immediate += q[j]
            load.append(load_immediate)
            route.append(j)

print(route, load)


