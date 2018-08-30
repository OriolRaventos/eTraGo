# -*- coding: utf-8 -*-
"""
"""
from config import sim_results_path, total_results_path, plot_path
# TODO: Make plot_path import work...don't know why it does not
from os import path, listdir

import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
import os
kmean = []


files1 = os.listdir(sim_results_path)

for c in range (1,len(files1)+1):

    path01= sim_results_path + 'ResultsExpansions' + str(c) +'.csv'
    Total = pd.read_csv(path01)
    k= Total.loc[0]['k-mean']
    
    if c == 1:
        results = Total
    else:
        results= results.append(Total)
 
    if (k in kmean) == False:
            kmean.append(k)

results.rename(columns={'Unnamed: 0':'Snapshots'}, inplace=True)
results.to_csv(total_results_path + 'TotalResults.csv')

for i in range(len(kmean)):
    value = int(kmean[i])
    RS = results.loc[results ['k-mean'] == value]
    RBMST = RS.loc[(RS['TypeSim'] == 'BM') & (RS['Storage'] == True)]
    RBMST = RBMST.sort_values(['Snapshots'], ascending = [1]) 
    RBMnoST = RS.loc[(RS['TypeSim'] == 'BM') & (RS['Storage'] == False)]
    RBMnoST = RBMnoST.sort_values(['Snapshots'], ascending = [1])
    RRemSnapshots = RS.loc[(RS['TypeSim'] == 'Remarkable Snapshots')]

    RRemSnapshots =  RRemSnapshots[RRemSnapshots.end_snapshot > 20]
    RRemSnapshots['timesteps'] = RRemSnapshots.end_snapshot - RRemSnapshots.start_snapshot
    RRemSnapshots['Simulation_time'] = RRemSnapshots['1st LOPF'] + RRemSnapshots['2nd LOPF']
    
    RBMnoST =  RBMnoST[RBMnoST.end_snapshot > 20]
    RBMnoST['timesteps'] = RBMnoST.end_snapshot - RBMnoST.start_snapshot
    RBMnoST['Simulation_time'] = RBMnoST['1st LOPF'] + RBMnoST['2nd LOPF']
    
    if not (RRemSnapshots.empty or RBMnoST.empty):
        fig, ax = plt.subplots()
        ax2 = ax
        RRemSnapshots.plot(x='timesteps', y='Simulation_time',\
                           title = 'simulation time for k-mean= '+ str(value),  label="2 LOPFs")
           
        RBMnoST.plot(x='timesteps', y='Simulation_time',\
                           title = 'simulation time for k-mean= '+ str(value),  label="BM")
        
        plt.show()
        
        
        fig = ax.get_figure()
        
        
    ##Plot graphs without storage
    """ if not (R2LOPFnoST.empty or RBMnoST.empty):
        fig, ax = plt.subplots()
        ax2 = ax 
        if not R2LOPFnoST.empty:
            R2LOPFnoST.plot(x='Snapshots', y='2nd LOPF', title = 'simulation time for k-mean= '+ str(value) + ' without Storage', ax=ax, label="2 LOPFs")
        if not RBMnoST.empty:
            RBMnoST.plot(x='Snapshots', y='2nd LOPF', title = 'simulation time for k-mean= '+ str(value) + ' without Storage', ax = ax2, label="Benchmark", ls = "--")
    
        ax.set_ylabel('Simulation Time 2nd LOPF (s)')
        ax.set_xlabel('Number of Snapshots')
        
        plt.show()
        
        fig = ax.get_figure()
        fig.savefig(path.join(plot_path, 'simulation time for k-mean= '+ str(value) +' without Storage.eps'))"""
        