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
H = list(range(0,600)) # the avaulable time (s), for now 1 hour
M=100000000000 #bigM
pta=35 #the loading/unloading time for agvs
ptf=60 #the loading/unlaoding time for forklifts

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


Predecessors=np.zeros((len(Tasks),1))
for n in Tasks:
    while Task_list.loc[n,'Predecessors'] != 0:
        Predecessors[n,0] = Task_list.loc[n,'Task ID']-Task_list.loc[n,'Predecessors']


'''
Intermediate_Matrix=pd.DataFrame(columns=['line add','stop add'])
Intermediate_Matrix['line add']=Task_list['End point line']-Task_list['Start point line']
Intermediate_Matrix['stop add']=Task_list['End point stop']-Task_list['Start point stop']
'''

model = Model ('AGV_Test')

#Define variables
xf={} # =1, if forklift f transports task n in time t is in point (l,s).
for l in Line:
    for s in Stop:
        for f in Forklifts:
            for n in Tasks:
                for t in H:
                    xf[l,s,f,n,t]=model.addVar( vtype=GRB.BINARY, name="xf["+str(l)+","+str(s)+","+str(f)+","+str(n)+","+str(t)+"]")

xa={} #=1, if AGV a transports task n in time t is in point (l,s).
for l in Line:
    for s in Stop:
        for a in AGVs:
            for n in Tasks:
                for t in H:
                    xa[l,s,a,n,t]=model.addVar( vtype=GRB.BINARY, name="xa["+str(l)+","+str(s)+","+str(a)+","+str(n)+","+str(t)+"]")

yf={} #=1, if task n is received by forklift f in time t
for f in Forklifts:
    for n in Tasks:
        for t in H:
            yf[f,n,t]=model.addVar ( vtype=GRB.BINARY, name="yf["+str(f)+","+str(n)+","+str(t)+"]")

ya={} #=1, if task n is received by AGV a in time t
for a in AGVs:
    for n in Tasks:
        for t in H:
            ya[a,n,t]=model.addVar ( vtype=GRB.BINARY, name="ya["+str(a)+","+str(n)+","+str(t)+"]")

#other intermediate variables
TF={}
for n in Tasks:
    TF[n]=model.addVar(vtype=GRB.CONTINUOUS)

TA={}
for n in Tasks:
    TA[n]=model.addVar(vtype=GRB.CONTINUOUS)

TT={}
TT=model.addVar(vtype=GRB.CONTINUOUS)

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

for n in Tasks:
    model.addConstr(quicksum(t * xf[Task_list.loc[n,'Start point line'],Task_list.loc[n,'Start point stop'],f,n,t] for t in H for f in Forklifts) <=quicksum(t * xf[Task_list.loc[n,'End point line'],Task_list.loc[n,'End point stop'],f,n,t] for t in H for f in Forklifts) + M*(1-Pickpoint[l,s,n]*Droppoint[l,s,n]))


for n in Tasks:
     model.addConstr(quicksum(t * xa[Task_list.loc[n,'Start point line'],Task_list.loc[n,'Start point stop'],a,n,t] for t in H for a in AGVs) <=quicksum(t * xa[Task_list.loc[n,'End point line'],Task_list.loc[n,'End point stop'],a,n,t] for t in H for a in AGVs) + M*(1-Pickpoint[l,s,n]*Droppoint[l,s,n]))

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
for l in Line:
    for s in Stop[1:-1]:
        for a in AGVs:
            for n in Tasks:
                for t in H:
                    for aa in AGVs:
                        for nn in Tasks:
                            model.addConstr(xa[l, s, a, n, t] + xa[l, s+1, aa, nn, t]+xa[l,s-1,a,n,t+0.5]+xa[l,s,aa,nn,t+0.5] <=3)
                            model.addConstr(xa[l, s, a, n, t] + xa[l, s-1, aa, nn, t] + xa[l, s+1, a, n, t + 0.5] + xa[l, s, aa, nn, t + 0.5] <= 3)
for l in Line:
    for s in Stop[1:-1]:
        for f in Forklifts:
            for n in Tasks:
                for t in H:
                    for ff in Forklifts:
                        for nn in Tasks:
                            model.addConstr(xf[l, s, f, n, t] + xf[l, s+1, ff, nn, t]+xf[l,s-1,f,n,t+0.33]+xf[l,s,ff,nn,t+0.33] <=3)
                            model.addConstr(xf[l, s, f, n, t] + xf[l, s-1, ff, nn, t] + xf[l, s+1, f, n, t + 0.33] + xf[l, s, ff, nn, t + 0.33] <= 3)

for l in Line:
    for s in Stop[1:-1]:
        for f in Forklifts:
            for n in Tasks:
                for t in H:
                    for a in AGVs:
                        for nn in Tasks:
                            model.addConstr(xa[l, s, a, n, t] + xf[l, s+1, f, nn, t]+xa[l,s-1,a,n,t+0.5]+xf[l,s,f,nn,t+0.33] <=3)
                            model.addConstr(xa[l, s, a, n, t] + xf[l, s-1, f, nn, t] + xa[l, s+1, a, n, t + 0.5] + xf[l, s, f, nn, t + 0.33] <= 3)

#Ensure that each AGV/Forklift cannot be located at more than one point at the same time.
for a in AGVs:
    for n in Tasks:
        for t in H:
            model.addConstr(quicksum(xa[l,s,a,n,t] for l in Line for s in Stop) <=1)

for f in Forklifts:
    for n in Tasks:
        for t in H:
            model.addConstr(quicksum(xf[l,s,f,n,t] for l in Line for s in Stop) <=1)

#How the vehicle walk is defined, at the next moment (the AGV/Forklift goes through one side of the grid), and AGV/Forklift cannot walk diagonally in oppositenodes.



#Ensure an AGV / a forklift waits for pt time unit (unloading/loading time unit) for task n in P/D point (i,j).
for l in Line:
    for s in Stop:
        for a in AGVs:
            for n in Tasks:
                for t in H:
                    model.addConstr(quicksum(xa[l,s,a,n,tt] for tt in range(t,t+pta)) >= (pta + 1)*xa[l,s,a,n,t]*(Pickpoint[l,s,n] + Droppoint[l,s,n]))
for l in Line:
    for s in Stop:
        for f in Forklifts:
            for n in Tasks:
                for t in H:
                    model.addConstr(quicksum(xf[l,s,f,n,tt] for tt in range(t,t+ptf)) >= (ptf + 1)*xf[l,s,f,n,t]*(Pickpoint[l,s,n] + Droppoint[l,s,n]))

#Defines the number of AGVs/forklifts on the map, allowing NumA/NumF AGVs/Forklifts to walk on the map at the same time
for n in Tasks:
    for t in H:
        model.addConstr(quicksum(xa[l,s,a,n,t] for a in AGVs for l in Line for s in Stop) <= len(AGVs))
        model.addConstr(quicksum(xf[l, s, f, n, t] for f in Forklifts for l in Line for s in Stop) <= len(Forklifts))

#Ensure each AGV can only take charge of the transferring task when the predecessor task taken by forklift has been finished
for l in Line:
    for s in Stop:
        for n in Tasks:
            for a in AGVs:
                for f in Forklifts:
                    model.addConstr(quicksum(t * xa[l,s,a,n+ Predecessors[n],t] for t in H)  >= quicksum(t * xf[l,s,f,n,t]))

#Ensure each AGV/forklift can be only responsible for transporting one task at each time unit.
for a in AGVs:
    for t in H:
        model.addConstr(quicksum(xa[l,s,a,n,t] for l in Line for s in Stop for n in Tasks) <=1)
for f in Forklifts:
    for t in H:
        model.addConstr(quicksum(xf[l,s,f,n,t] for l in Line for s in Stop for n in Tasks) <=1)

#------Scheduling constraints------

#Ensures that a task is assigned to an AGV/Forklift when that AGV, Forklift is in starting point.
for a in AGVs:
    for n in Tasks:
        for t in H:
            model.addConstr(ya[a,n,t] <= xa[0,225,a,n,t])
for f in Forklifts:
    for n in Tasks:
        for t in H:
            model.addConstr(yf[f,n,t] <= xf[0,0,f,n,t])

#Ensure each AGV/forklift should receive a task at the time 0 certainly.
for a in AGVs:
    model.addConstr(quicksum(ya[a,n,0] for n in Tasks) == 1)
for f in Forklifts:
    model.addConstr(quicksum(yf[f,n,0] for n in Tasks) == 1)

#Ensure that each task can be assigned to only one AGV or one Forklift
for n in Tasks:
    model.addConstr(quicksum(ya[a,n,t] for t in H for a in AGVs) + quicksum(yf[f,n,t] for t in H for f in Forklifts) == 1)

#Ensure that when a new task can be assigned to an AGV/Forklift after that a task had been completed by same AGV at last time unit
for a in AGVs:
    for n in Tasks:
        for t in H:
            model.addConstr(ya[a,n,t] <= quicksum(xa[l,s,a,nn,t-1]*Droppoint[l,s,nn] for l in Line for s in Stop for nn in Tasks))
for f in Forklifts:
    for n in Tasks:
        for t in H:
            model.addConstr(yf[f,n,t] <= quicksum(xf[l,s,f,nn,t-1]*Droppoint[l,s,nn] for l in Line for s in Stop for nn in Tasks))


#Ensure each AGV can only take charge of the transferring task when the predecessor task taken by forklift has been finished, so the AGV arrive time of the pick point should no earlier than the Forklift delivery time.
for a in AGVs:
    for f in Forklifts:
        for n in Tasks:
            model.addConstr(quicksum(t*ya[a, n+ Predecessors[n],t] for t in H) >= quicksum(t*yf[f,n,t] for t in H))








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
