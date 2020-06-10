import pypsa
import pandas as pd
import numpy as np
from numpy import genfromtxt
import time
import datetime
import os

if 'READTHEDOCS' not in os.environ:
    # Sphinx does not run this code.
    # Do not import internal packages directly
    from etrago.tools.io import (
        NetworkScenario,
        results_to_oedb,
        extension,
        decommissioning)
    from etrago.tools.plot import (
        plot_line_loading,
        plot_stacked_gen,
        add_coordinates,
        curtailment,
        gen_dist,
        storage_distribution,
        storage_expansion,
        extension_overlay_network,
        nodal_gen_dispatch)

    from etrago.tools.utilities import (
        load_shedding,
        data_manipulation_sh,
        convert_capital_costs,
        results_to_csv,
        parallelisation,
        pf_post_lopf,
        loading_minimization,
        calc_line_losses,
        group_parallel_lines)
    from etrago.tools.extendable import extendable
    from etrago.cluster.networkclustering import (
        busmap_from_psql, cluster_on_extra_high_voltage, kmean_clustering)
    from etrago.cluster.snapshot_adjacent import snapshot_clustering
    #from egoio.tools import db #URI: Sudently gave a massive error after the shutdown on 7 Sept. 2018
    from sqlalchemy.orm import sessionmaker


args = {  # Setup and Configuration:
    'db': 'esa',  # database session
    'gridversion': 'v0.2.11',  # None for model_draft or Version number
    'method': 'lopf',  # lopf or pf
    'pf_post_lopf': False,  # perform a pf after a lopf simulation
    'start_snapshot': 1,
    'end_snapshot': 8760,
    'solver': 'gurobi',  # glpk, cplex or gurobi
    'solver_options': {
            'Method': 2, #changed to Barrier
            'threads': 4, 
            'LogFile': '/home/raventos/Results_SHNEP2035_k100mod1/Gurobi.log' ,
            'OutputFlag': 1
            },  # {} for default or dict of solver options
    'scn_name': 'SH NEP 2035',  # a scenario: Status Quo, NEP 2035, eGo100
    # Scenario variations:
    'scn_extension': None,  # None or extension scenario
    'scn_decommissioning': None,  # None or decommissioning scenario
    'add_Belgium_Norway': False,  # add Belgium and Norway
    # Export options:
    'lpfile': False,  # save pyomo's lp file: False or /path/tofolder
    'results': '/home/raventos/Results_SHNEP2035_k100mod1/',  # save results as csv: False or /path/tofolder
    'export': False,  # export the results back to the oedb
    # Settings:
    'extendable': ['storages'],  # None or array of components to optimize
    'generator_noise': False,  # apply generator noise, False or seed number
    'minimize_loading': False,
    # Clustering:
    'network_clustering_kmeans': 10,  # False or the value k for clustering
    'load_cluster': False,  # False or predefined busmap for k-means
    'network_clustering_ehv': False,  # clustering of HV buses to EHV buses.
    'snapshot_clustering': False,  # False or the number of 'periods'
    # Simplifications:
    'parallelisation': False,  # run snapshots parallely.
    'skip_snapshots': False,
    'line_grouping': False,  # group lines parallel lines
    'branch_capacity_factor': 0.7,  # factor to change branch capacities
    'load_shedding': False,  # meet the demand at very high cost
    'comments': None}



#adjust the path to pypsa examples directory
main_folder = '/home/raventos/paper/200nodesNEP2035v045/'
network = pypsa.Network(csv_folder_name= main_folder + 'Data/')

"""
#print and plot things

pd.set_option("display.max_rows", 100)
pd.set_option("display.max_columns", 100)
pd.set_option("display.max_colwidth", 100)
pd.set_option("display.width", None)

print(network.generators.p_nom_opt)

print(network.generators_t.p)

print(network.storage_units.p_nom_opt)

print(network.storage_units_t.p)

print(network.lines.s_nom_opt)

print(network.lines_t.p0)

network.plot()
"""

#Here we do the Time Aggregation Series

network_aux=network
mylist = [5,10,20,30,40,50,60,70,80,90,100,150,200,250,300,350,365] 
mylist = [a*24 for a in mylist]
for count in mylist: 
    
    args['snapshot_clustering']=count
    network=network_aux
    ###rename folder:
    clusterMethod = 'chronological' #'hierarchical', 'k_means', 'k_medoids', 'hierarchicalwithpeaks'
    directory = main_folder + clusterMethod + '/' + clusterMethod + str(count)
    if not os.path.exists(directory):
        os.makedirs(directory)
    else: 
        directory = directory + 'copy'
        os.makedirs(directory)
    args['results']=directory

    # snapshot clustering
    #URI: the clusterMethod was added to get more flexibility
    #it doesn't exists int the original snapshot.py in etrago
    network, snapshot_map, day_map = snapshot_clustering(
            network, how='hourly', clusters=args['snapshot_clustering'],clusterMethod = clusterMethod, normed=True)
    #base = np.ones(24, dtype=int)
    #periods = base*0
    #newcount=len(np.unique(day_map))
    #for i in range(1,newcount):
    #    periods = np.concatenate([periods,base*i])
    extra_functionality = None  # daily_bounds ,couple_design_periods, decouple_design_periods, inter_intra_soc

    # start linear optimal powerflow calculations
    x = time.time()    
    try:
        network.lopf(
                network.snapshots,
                formulation='angles',
                solver_name='gurobi',
                solver_options =  {
                    'Method': 2,  #changed to Barrier
                    'threads': 4, 
                    'LogFile': directory + '/gurobi.log',  
                    'OutputFlag': 1,
                    'BarConvTol' : 1.e-4, #1.e-12 ###1e-8 # [0..1]
                    'Crossover' : 0,   # or -1
                    'FeasibilityTol' : 1e-4, ###1e-6 # [1e-9..1e-2]
                    'BarHomogeneous' : 1,
                    'Presolve' : 2
                    },
                extra_functionality=extra_functionality) # either None or extra_functionality
                #URI: this equals soc initial and final for each typ day / could put = None
    except Exception:
        print("Numerical trouble encountered, but we continue anyway" + "\n")
    y = time.time()
    z = (y - x) / 60
    # z is time for lopf in minutes
    print("Time for LOPF [min]:", round(z, 2))

    # provide storage installation costs
    if not network.storage_units.p_nom_opt.isnull().values.any():
        if sum(network.storage_units.p_nom_opt) != 0:
            installed_storages = \
                network.storage_units[network.storage_units.p_nom_opt != 0]
            storage_costs = sum(    
                installed_storages.capital_cost *
                installed_storages.p_nom_opt)
            print(
                "Investment costs for all storages in selected snapshots [EUR]:",
                round(
                    storage_costs,
                    2))
    
    #write the csv files
    try:
        results_to_csv(network,args)
    except Exception:
        print("\n" + "So there is really nothing to export")
   

