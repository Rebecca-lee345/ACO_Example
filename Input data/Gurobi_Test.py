#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 20:12:46 2022

@author: lichen
"""

import pandas as pd
import numpy as np
from gurobipy import *

#Input task list
Task_list = pd.read_excel('Task_list_Test.xlsx', sheet_name='Tasklist', usecols="A:U",skiprows=0, nrows=10)


#Parameters definition
Line = list(range(0,3)) #the number of lines in the roadmap
Stop = list(range(0,450)) #the number of stops in each line
Forklifts = list(range(0,2)) #the number of forklifts
AGVs = list(range(0,4)) #teh number of AGVs
Tasks = Task_list['Task ID'].tolist() #the task ID
H = list(range(0,3600)) # the avaulable time (s), for now 1 hour
M=100000000000 #bigM

# Parameter creation
Droppoint=np.zeros((len(Line), len(Stop),len(Tasks) ))
for l in Line:
    for s in Stop:
        for n in Tasks:
            if Task_list.loc[n,'End point line'] == l and Task_list.loc[n,'End point stop'] == s :
                Droppoint[l, s, n] = 1

Pickpoint=np.zeros((len(Line), len(Stop),len(Tasks) ))
for l in Line:
    for s in Stop:
        for n in Tasks:
            if Task_list.loc[n,'Start point line'] == l and Task_list.loc[n,'Start point stop'] == s :
                Pickpoint[l, s, n] = 1

Intermediate_Matrix=pd.DataFrame(columns=['line add','stop add'])
Intermediate_Matrix['line add']=Task_list['End point line']-Task_list['Start point line']
Intermediate_Matrix['stop add']=Task_list['End point stop']-Task_list['Start point stop']


model = Model ('AGV_Test')

#Define variables
xf={} # =1, if forklift f transports task n in time t is in point (l,s).
for l in Line:
    for s in Stop:
        for f in Forklifts:
            for n in Tasks:
                for t in H:
                    xf[l,s,f,n,t]=model.addVar( vtype=GRB.BINARY, name="x["+str(l)+","+str(s)+","+str(f)+","+str(n)+","+str(t)+"]")

xa={} #=1, if AGV a transports task n in time t is in point (l,s).
for l in Line:
    for s in Stop:
        for a in AGVs:
            for n in Tasks:
                for t in H:
                    xa[l,s,a,n,t]=model.addVar( vtype=GRB.BINARY, name="x["+str(l)+","+str(s)+","+str(a)+","+str(n)+","+str(t)+"]")

yf={} #=1, if task n is received by forklift f in time t
for f in Forklifts:
    for n in Tasks:
        for t in H:
            yf[f,n,t]=model.addVar ( vtype=GRB.BINARY, name="x["+str(f)+","+str(n)+","+str(t)+"]")

yf={} #=1, if task n is received by AGV a in time t
for a in AGVs:
    for n in Tasks:
        for t in H:
            yf[a,n,t]=model.addVar ( vtype=GRB.BINARY, name="x["+str(a)+","+str(n)+","+str(t)+"]")

#other intermediate variables
TF={}
for n in Tasks:
    TF[n]=model.addVar(vtype=GRB.CONTINUOUS)

TA={}
for n in Tasks:
    TA[n]=model.addVar(vtype=GRB.CONTINUOUS)
TT=[]

model.update()

#Objective function
model.setObjective(TT)
model.modelSense = GRB.MINIMIZE
model.update()

#Constraints
#Makespan calculation
for n in Tasks:
    model.addConstr(TT>=TF[n])
for n in Tasks:
    model.addConstr(TT>=TA[n])

for n in Tasks:
    model.addConstr(TF[n]== quicksum(t * xf[l,s,f,n,t]*Droppoint[l,s,n] for l in Line for s in Stop for f in Forklifts for t in H))
for n in Tasks:
    model.addConstr(TA[n]== quicksum(t * xa[l,s,a,n,t]*Droppoint[l,s,n] for l in Line for s in Stop for a in AGVs for t in H))

#Route planning
#Each required P/D point for each task n should be visited by either a forklift or AGV.
for l in Line:
    for s in Stop:
        for n in Tasks:
            model.addConstr(quicksum(xf[l,s,f,n,t] for t in H for f in Forklifts) + quicksum(xa[l,s,a,n,t] for t in H for a in AGVs) >= Pickpoint[l,s,n] + Droppoint[l,s,n])

#Ensure if point (l,s) is the pick point which precedes drop point (l’,s’) for task n, thus arrive time of AGV/Forklift to (l,s) is earlier than arrive time (l’,s’)
for l in Line:
    for s in Stop:
        for n in Tasks:
            model.addConstr(quicksum(t * xf[l,s,f,n,t] for t in H for f in Forklifts) <=quicksum(t * xf[l+Intermediate_Matrix[n,'line add'],s+Intermediate_Matrix[n,'stop add'],f,n,t] for t in H for f in Forklifts) + M*(1-Pickpoint[l,s,n]*Droppoint[l,s,n]))
for l in Line:
    for s in Stop:
        for n in Tasks:
            model.addConstr(quicksum(t * xa[l,s,a,n,t] for t in H for a in AGVs) <=quicksum(t * xa[l+Intermediate_Matrix[n,'line add'],s+Intermediate_Matrix[n,'stop add'],a,n,t] for t in H for a in AGVs) + M*(1-Pickpoint[l,s,n]*Droppoint[l,s,n]))

#At the same time, it can only be one AGV or one Forklift on each point of the shared route except the starting point. (Guarantee conflict-free route)
for l in Line:
    for s in Stop:
        for t in H:
            model.addConstr(quicksum(xf[l,s,f,n,t] for f in Forklifts for n in Tasks) + quicksum(xa[l,s,a,n,t] for a in AGVs for n in Tasks) <=1)

#At the same time, it can only be one AGV on each point of the network except the starting point.(Guarantee conflict-free route)
for l in Line:
    for s in Stop:
        for t in H:
            model.addConstr(quicksum(xa[l,s,a,n,t] for a in AGVs for n in Tasks) <=1)

#Ensure that AGVs/forklifts have no conflicts on the horizontal and vertical edges of the network.∆t/∆t’ is the time taken by the AGV/Forklift to walk the side length of the grid. The equation defines the collision situation that may occur at the edge of the map.
















model.update()
model.setParam( 'OutputFlag', True)
#model.setParam ('MIPGap', 0);
model.optimize()
#model.computeIIS()
#model.write('model_problem.ilp')
status = model.status
#model.write("output.lp")

if status == GRB.Status.UNBOUNDED:
    print('The model cannot be solved because it is unbounded')

elif status == GRB.Status.OPTIMAL or True:
    f_objective = model.objVal
    print('***** RESULTS ******')
    print('\nObjective Function Value: \t %g' % f_objective)

elif status != GRB.Status.INF_OR_UNBD and status != GRB.Status.INFEASIBLE:
    print('Optimization was stopped with status %d' % status)