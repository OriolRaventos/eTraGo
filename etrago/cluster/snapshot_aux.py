# -*- coding: utf-8 -*-
# Copyright 2016-2018  Flensburg University of Applied Sciences,
# Europa-Universität Flensburg,
# Centre for Sustainable Energy Systems


# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation; either version 3 of the
# License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# File description for read-the-docs
""" This module contains functions for calculating representative days/weeks
based on a pyPSA network object. It is designed to be used for the `lopf`
method. Essentially the tsam package
( https://github.com/FZJ-IEK3-VSA/tsam ), which is developed by 
Leander Kotzur is used.

Remaining questions/tasks:

- Does it makes sense to cluster normed values?
- Include scaling method for yearly sums
"""

import pandas as pd
import pyomo.environ as po
import tsam.timeseriesaggregation as tsam
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

import pypsa
from pyomo.environ import *
from pypsa.opt import LExpression, LConstraint, l_constraint
from pypsa.descriptors import get_switchable_as_dense



__copyright__ = ("Flensburg University of Applied Sciences, "
                 "Europa-Universität Flensburg, "
                 "Centre for Sustainable Energy Systems")
__license__ = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__author__ = "Simon Hilpert"


def snapshot_clustering(network, how='daily', clusters=10, clusterMethod = 'hierarchical',normed=False):

    network, snapshot_map, day_map = run(network=network.copy(), n_clusters=clusters,
                  how=how, normed=normed, clusterMethod = clusterMethod)
    return network, snapshot_map, day_map


def tsam_cluster(timeseries_df, typical_periods=10, how='daily', clusterMethod = 'hierarchical'):
    """
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with timeseries to cluster
    
    clusterMethod : 'hierarchical', 'k_means', 'k_medoids', 'hierarchicalwithpeaks'

    Returns
    -------
    timeseries : pd.DataFrame
        Clustered timeseries
    """

    if how == 'daily':
        hours = 24
    if how == 'weekly':
        hours = 168
    
    extremePeriodMethod = 'None'
    clusterMethodadd = ''
    
    ###########################################################    
    #URI: This adds peaks
    peakloadcol = '0'
    peakwindcol = '0 wind'
    peaksolarcol = '0 solar'
    if clusterMethod == 'hierarchicalwithpeaks':
    
        #URI: We find the day with more peaks and pick one of the LOADS with such.
        #Get indices of the maximums
        loading = timeseries_df.filter(regex='0$|1$|2$|3$|4$|5$|6$|7$|8$|9$',axis=1)
        loading_max = loading.idxmax()
        #Get rid of the hours
        #loading_aux = loading.apply(lambda x: str(x.year) + "-" + str(x.month) + "-" + str(x.day))
        loading_aux=loading_max.apply(lambda x: x.strftime("%y-%m-%d"))
        #find the day that is peak for most of the loads
        #Notice: We just pick the fist date, could be other days with the same maximums
        #This comes from: https://stackoverflow.com/questions/6987285/python-find-the-item-with-maximum-occurrences-in-a-list
        from collections import defaultdict
        from operator import itemgetter
        c = defaultdict(int)
        for i in loading_aux:
            c[i] += 1
        peakloadday=max(c.items(), key=itemgetter(1))
        for idx, val in enumerate(loading_aux):
            if val == peakloadday[0]:
                peakloadcol = loading_aux.index.values[idx]
                break
        
        
        #URI: We find the day with more peaks and pick one of the WIND generators with such.
        #Get indices of the maximums
        wind = timeseries_df.filter(regex='wind$', axis=1)
        wind_max = wind.idxmax()
        #Get rid of the hours
        #loading_aux = loading.apply(lambda x: str(x.year) + "-" + str(x.month) + "-" + str(x.day))
        wind_aux=wind_max.apply(lambda x: x.strftime("%y-%m-%d"))
        #find the day that is peak for most of the loads
        #Notice: We just pick the fist date, could be other days with the same maximums
        #This comes from: https://stackoverflow.com/questions/6987285/python-find-the-item-with-maximum-occurrences-in-a-list
        from collections import defaultdict
        from operator import itemgetter
        c = defaultdict(int)
        for i in wind_aux:
            c[i] += 1
        peakwindday=max(c.items(), key=itemgetter(1))
        for idx, val in enumerate(wind_aux):
            if val == peakwindday[0]:
                peakwindcol = wind_aux.index.values[idx]
                break
        
        
        
        #URI: We find the day with more peaks and pick one of the SOLAR generators with such.
        #Get indices of the maximums
        solar = timeseries_df.filter(regex='solar$', axis=1)
        solar_max = solar.idxmax()
        #Get rid of the hours
        #loading_aux = loading.apply(lambda x: str(x.year) + "-" + str(x.month) + "-" + str(x.day))
        solar_aux=solar_max.apply(lambda x: x.strftime("%y-%m-%d"))
        #find the day that is peak for most of the loads
        #Notice: We just pick the fist date, could be other days with the same maximums
        #This comes from: https://stackoverflow.com/questions/6987285/python-find-the-item-with-maximum-occurrences-in-a-list
        from collections import defaultdict
        from operator import itemgetter
        c = defaultdict(int)
        for i in solar_aux:
            c[i] += 1
        peaksolarday=max(c.items(), key=itemgetter(1))
        for idx, val in enumerate(solar_aux):
            if val == peaksolarday[0]:
                #This is not the column with the overall maximum, just the first column with the maximum at the place
                #in time with more maximums
                peaksolarcol = solar_aux.index.values[idx]
                break
                         
        clusterMethod = 'hierarchical'
        clusterMethodadd = 'withpeaks'
        extremePeriodMethod = 'new_cluster_center'
     
    
    ##################################################################
    #URI: This is the original method=hierarchical
    if clusterMethod == 'hierarchical':
        aggregation = tsam.TimeSeriesAggregation(
            timeseries_df,
            noTypicalPeriods=typical_periods,
            rescaleClusterPeriods=False,
            hoursPerPeriod=hours,
            clusterMethod= clusterMethod, #averaging, k_means, k_medoids, hierarchical
            extremePeriodMethod = extremePeriodMethod , #'None', 'append', 'new_cluster_center', 'replace_cluster_center'
            addPeakMax = [peakloadcol,peakwindcol,peaksolarcol])
        
        timeseries = aggregation.createTypicalPeriods()
        #URI: Better take the whole thing
        timeseries_new =aggregation.predictOriginalData()
        cluster_weights = aggregation.clusterPeriodNoOccur
    
        
        # get the medoids/ the clusterCenterIndices
        clusterCenterIndices = aggregation.clusterCenterIndices
        
        #URI: and add the peak periods
        if not extremePeriodMethod is 'None':
            clusterOrder = aggregation.clusterOrder
            for i in range(len(clusterCenterIndices),len(cluster_weights.keys())):
                clusterCenterIndices.append(np.where(clusterOrder == i)[0][0]) 
        
        # get all index for every hour of that day of the clusterCenterIndices
        start = []
        # get the first hour of the clusterCenterIndices (days start with 0)
        for i in clusterCenterIndices:
            start.append(i * hours)
    
        # get a list with all hours belonging to the clusterCenterIndices
        nrhours = []
        for j in start:
            nrhours.append(j)
            x = 1
            while x < hours:
                j = j + 1
                nrhours.append(j)
                x = x + 1
    
        # get the origial Datetimeindex
        dates = timeseries_df.iloc[nrhours].index
    
    
    ######################################################
    #URI: This is them method=k_means or k_medoids
    elif clusterMethod == 'k_means' or clusterMethod == 'k_medoids':
        aggregation = tsam.TimeSeriesAggregation(
            timeseries_df,
            noTypicalPeriods=typical_periods,
            rescaleClusterPeriods=False,
            hoursPerPeriod=hours,
            clusterMethod=clusterMethod)#averaging, k_means, k_medoids, hierarchical
            #solver='gurobi' #This was made available in version 1.0.0
    
        timeseries = aggregation.createTypicalPeriods()
        #URI: Better take the whole thing
        timeseries_new =aggregation.predictOriginalData()
        cluster_weights = aggregation.clusterPeriodNoOccur
    
        
        # get a representative asclusterCenterIndices
        #clusterCenterIndices = aggregation.clusterCenterIndices
        clusterCenterIndices = []
        for i in cluster_weights.keys():
            clusterCenterIndices.append(np.argmax(aggregation.clusterOrder == i))
        
        #URI: and add the peak periods
        #clusterOrder = aggregation.clusterOrder
        #for i in range(len(clusterCenterIndices),len(cluster_weights.keys())):
        #    clusterCenterIndices.append(np.where(clusterOrder == i)[0][0]) 
        
        # get all index for every hour of that day of the clusterCenterIndices
        start = []
        # get the first hour of the clusterCenterIndices (days start with 0)
        for i in clusterCenterIndices:
            start.append(i * hours)
    
        # get a list with all hours belonging to the clusterCenterIndices
        nrhours = []
        for j in start:
            nrhours.append(j)
            x = 1
            while x < hours:
                j = j + 1
                nrhours.append(j)
                x = x + 1
    
        # get the origial Datetimeindex
        dates = timeseries_df.iloc[nrhours].index
    

    ##########################################################
    #URI: this is to get the data as csv
    directory = '/home/raventos/Auxiliary/tsamdata' + clusterMethod + clusterMethodadd + '/'
    timeseries_df.to_csv(directory + 'rawPeriods' + str(typical_periods) + '.csv')
    timeseries.to_csv(directory + 'typPeriods_unscaled' + str(typical_periods) + '.csv')
    timeseries_new.to_csv(directory + 'predictedPeriods_unscaled' + str(typical_periods) + '.csv')
    indexMatching=aggregation.indexMatching()
    indexMatching.to_csv(directory + 'indexMatching' + str(typical_periods) + '.csv')
    clusterOrder = aggregation.clusterOrder.astype(int)
    np.savetxt(directory + 'clusterOrder' + str(typical_periods) + '.csv', clusterOrder, fmt='%i', delimiter=",")
    cCI=pd.DataFrame(clusterCenterIndices,columns=['index'])
    cCI.to_csv(directory + 'clusterCenterIndices' + str(typical_periods) + '.csv')
    noOccur=pd.DataFrame(cluster_weights, index=['noOccur'])
    noOccur.to_csv(directory + 'noOccurrances' + str(typical_periods) + '.csv')
    aggregation.accuracyIndicators().to_csv(directory + 'indicators_unscaled' + str(typical_periods) + '.csv')
    np.savetxt(directory + 'dates' + str(typical_periods) + '.csv',dates.strftime("%Y-%m-%d %X"), fmt="%s", delimiter=",")

    #URI: This is for Bruno's coupling
    #dates = pd.read_csv(main_folder + method + str(count) + '/tsamdata/dates' + str(count) +'.csv', index_col = 0)
    dates2 =[]
    for row in indexMatching.index:
        dates2.append(dates[indexMatching.loc[row,'PeriodNum']*24 + indexMatching.loc[row,'TimeStep']])
    snapshot_map= pd.Series(dates2, index=indexMatching.index)
    
    #URI: Just checking
    snapshot_map.to_csv(directory + 'snapshot_map' + str(typical_periods) + '.csv')
    #print([peakloadcol,peakwindcol,peaksolarcol])
    #print(clusterCenterIndices)
    #print([peakloadday,peakwindday,peaksolarday])
    #print(dates)
    #print(cluster_weights)
    #print(aggregation.indexMatching())
    return timeseries_new, cluster_weights, dates, hours, snapshot_map, clusterOrder


def run(network, n_clusters=None, how='daily',
        normed=False, clusterMethod = 'hierarchical'):
    """
    """
    # reduce storage costs due to clusters
    #URI: This made nothing
    #network.cluster = True

    # calculate clusters
    tsam_pre, divisor= prepare_pypsa_timeseries(network, normed)
    tsam_ts, cluster_weights, dates, hours, snapshot_map, day_map = tsam_cluster(
            tsam_pre, typical_periods=n_clusters,
            how=how, clusterMethod = clusterMethod)
    #URI: It used to be (wrong): how = 'daily'
    
    #URI: Here we do the scaling and get the scaled data
    
    tsam_ts = rescaleData(tsam_pre,tsam_ts,divisor)
    directory = '/home/raventos/Auxiliary/tsamdata' + clusterMethod +'/'
    tsam_ts.to_csv(directory + 'predictedPeriods' + str(n_clusters) + '.csv')
    typPeriods = tsam_ts.loc[dates]
    typPeriods.to_csv(directory + 'typPeriods' + str(n_clusters) + '.csv')
    accIndicators = accuracyIndicators(tsam_ts,tsam_pre)
    accIndicators.to_csv(directory + 'indicators' + str(n_clusters) + '.csv')
    

    update_data_frames(network, tsam_ts, divisor, cluster_weights, dates, hours, normed)

    return network, snapshot_map, day_map


def prepare_pypsa_timeseries(network, normed=False):
    """
    """

    if normed:
        normed_loads = network.loads_t.p_set / network.loads_t.p_set.max()
        normed_renewables = network.generators_t.p_max_pu
        #URI: This divisor will simplify the update
        divisor= len(normed_renewables.columns)
        df = pd.concat([normed_renewables, normed_loads], axis=1)
    else:
        loads = network.loads_t.p_set
        renewables = network.generators_t.p_max_pu #URI:previously .p_set
        #URI: This divisor will simplify the update
        divisor= len(renewables.columns)
        df = pd.concat([renewables, loads], axis=1)

    return df, divisor


def update_data_frames(network, tsam_ts, divisor, cluster_weights, dates, hours, normed=False):
    """ Updates the snapshots, snapshots weights and the dataframes based on
    the original data in the network and the medoids created by clustering
    these original data.

    Parameters
    -----------
    network : pyPSA network object
    cluster_weights: dictionary
    dates: Datetimeindex


    Returns
    -------
    network

    """
    network.snapshot_weightings = network.snapshot_weightings.loc[dates]
    network.snapshots = network.snapshot_weightings.index

    # set new snapshot weights from cluster_weights
    snapshot_weightings = []
    for i in cluster_weights.values():
        x = 0
        while x < hours:
            snapshot_weightings.append(i)
            x += 1
    for i in range(len(network.snapshot_weightings)):
        network.snapshot_weightings[i] = snapshot_weightings[i]

    # put the snapshot in the right order
    #network.snapshots.sort_values() 
    #network.snapshot_weightings.sort_index() 
    #URI: If we want it to really work: BUT WE DON'T SINCE WE USE day_map
    #network.snapshots=network.snapshots.sort_values()
    #network.snapshot_weightings=network.snapshot_weightings.sort_index()  
    
    #URI: Need to separate generators from load
    #URI: and add the normed case
    if normed:
            network.generators_t.p_max_pu = tsam_ts.iloc[:,:divisor] #URI:previously .p_set
            network.loads_t.p_set = tsam_ts.iloc[:,divisor:]*network.loads_t.p_set.max()
    else:
        network.generators_t.p_max_pu = tsam_ts.iloc[:,:divisor] #URI:previously .p_set
        network.loads_t.p_set = tsam_ts.iloc[:,divisor:] 
    
    #URI: We don't want to keep numbers that are too small    
    network.generators_t.p_max_pu.where(lambda df: df>0.000001, other=0., inplace=True)
    network.loads_t.p_set.where(lambda df: df>0.000001, other=0., inplace=True)
    
    return network


def daily_bounds(network, snapshots):
    """ This will bound the storage level to 0.5 max_level every 24th hour.
    """
    
    sus = network.storage_units
    # take every first hour of the clustered days
    network.model.period_starts = network.snapshot_weightings.index[0::24]

    network.model.storages = sus.index

    def day_rule(m, s, p):
        """
        Sets the soc of the every first hour to the soc of the last hour
        of the day (i.e. + 23 hours)
        """
        return (
            m.state_of_charge[s, p] ==
            m.state_of_charge[s, p + pd.Timedelta(hours=23)])

    network.model.period_bound = po.Constraint(
        network.model.storages, network.model.period_starts, rule=day_rule)


####################################
def manipulate_storage_invest(network, costs=None, wacc=0.05, lifetime=15):
    # default: 4500 € / MW, high 300 €/MW
    crf = (1 / wacc) - (wacc / ((1 + wacc) ** lifetime))
    network.storage_units.capital_cost = costs / crf


def write_lpfile(network=None, path=None):
    network.model.write(path,
                        io_options={'symbolic_solver_labels': True})


def fix_storage_capacity(network, resultspath, n_clusters):  # "network" added
    path = resultspath.strip('daily')
    values = pd.read_csv(path + 'storage_capacity.csv')[n_clusters].values
    network.storage_units.p_nom_max = values
    network.storage_units.p_nom_min = values
    resultspath = 'compare-' + resultspath
    
def rescaleData(tsam_pre,tsam_ts,divisor):
    
    #first for the renewables
    for i in range(divisor):
        diff = 1.
        a=0
        while diff > 0.000001 and a < 20:
            scal = tsam_pre.iloc[:,i].sum()/tsam_ts.iloc[:,i].sum()
            k=(tsam_ts.iloc[:,i]>1.).sum()
            if scal > 1.0 and k > 0:
                #print(scal, k)
                scal = scal + k*(scal-1)
            tsam_ts.iloc[:,i]= tsam_ts.iloc[:,i]*scal
            diff = tsam_ts.iloc[:,i].max() - 1.
            for ii in range(len(tsam_ts.index)):
                if(tsam_ts.iloc[ii,i] > 1.): 
                    tsam_ts.iloc[ii,i]=1.
            a = a+1
        if a== 20:
            print ('Column ' + str(tsam_ts.columns[i]) + ' could not be scaled in 20 itereations')
            #print(diff)
            #print(tsam_ts.iloc[:,i].idxmax())
    
    #then for the load
    for i in range(divisor,len(tsam_pre.columns)):
        scal = tsam_pre.iloc[:,i].sum()/tsam_ts.iloc[:,i].sum()
        tsam_ts.iloc[:,i]=tsam_ts.iloc[:,i]*scal
    
    return tsam_ts

#This is taken from the tsam package by Leander Kotzur, but we want to use it here after the scaling
def accuracyIndicators(tsam_ts,tsam_pre):
    indicatorRaw = {
        'RMSE': {},
        'RMSE_duration': {},
        'MAE': {}}

    for column in tsam_pre.columns:
        origTS = tsam_pre[column]
        predTS = tsam_ts[column]
        indicatorRaw['RMSE'][column] = np.sqrt(
            mean_squared_error(origTS, predTS))
        indicatorRaw['RMSE_duration'][column] = np.sqrt(mean_squared_error(
            origTS.sort_values(ascending=False).reset_index(drop=True),
            predTS.sort_values(ascending=False).reset_index(drop=True)))
        indicatorRaw['MAE'][column] = mean_absolute_error(origTS, predTS)

    return pd.DataFrame(indicatorRaw)

########################################################################
#Bruno's Codes
########################################################################
    
### function to combine several different 'extra_functionalities' to be passed to pypsa.network.lopf()
def apply_funcs(*funcs):
    def applier(network, snapshots):
        for func in funcs:
            func(network, snapshots)
    return applier

### function to modify the soc constraint of pypsa.network.lopf()
### in order to couple design periods according to the methodology of Gabrielli et al. [2017]
### possibly also includes an overall line capacity cap constraint
### returns a function to be passed as extra_functionality to pypsa.network.lopf()
def couple_design_periods(all_snapshots, snapshot_map, line_options=None):

    def extra_functionality(network, snapshots):

        model = network.model

        sus = network.storage_units

        ext_sus_i = sus.index[sus.p_nom_extendable]
        fix_sus_i = sus.index[~ sus.p_nom_extendable]

        inflow = get_switchable_as_dense(network, 'StorageUnit', 'inflow', snapshots)
        spill_sus_i = sus.index[inflow.max()>0] #skip storage units without any inflow
        inflow_gt0_b = inflow>0
        spill_bounds = {(su,sn) : (0,inflow.at[sn,su])
                    for su in spill_sus_i
                    for sn in snapshots
                    if inflow_gt0_b.at[sn,su]}
        spill_index = spill_bounds.keys()

        ### define new state of charge variables
        model.extra_state_of_charge = Var(list(network.storage_units.index), all_snapshots,
                                    domain=NonNegativeReals, bounds=(0,None))

        upper = {(su,sn) : [[(1,model.extra_state_of_charge[su,sn]),
                     (-sus.at[su,"max_hours"],model.storage_p_nom[su])],"<=",0.]
                 for su in ext_sus_i for sn in all_snapshots}
        upper.update({(su,sn) : [[(1,model.extra_state_of_charge[su,sn])],"<=",
                         sus.at[su,"max_hours"]*sus.at[su,"p_nom"]]
                  for su in fix_sus_i for sn in all_snapshots})

        l_constraint(model, "extra_state_of_charge_upper", upper,
             list(network.storage_units.index), all_snapshots)
        
        
        ####update state_of_charge to get the data in Pypsa (just for "snapshots", not defined for "all_snapshots")
        #updatesoc = {(su,sn) : [[(1,model.extra_state_of_charge[su,sn]),
        #             (-1,model.state_of_charge[su,sn])],"==",0.]
        #         for su in sus.index for sn in snapshots}    
        #l_constraint(model, "update_soc", updatesoc,
        #     list(network.storage_units.index), snapshots)
        


        ### define soc constraint
        soc = {}

        state_of_charge_set = get_switchable_as_dense(network, 'StorageUnit', 'state_of_charge_set', snapshots)

        for su in sus.index:
                for i,sn in enumerate(all_snapshots):

                    soc[su,sn] =  [[],"==",0.]

                    elapsed_hours = 1

                    design_sn = snapshot_map[sn]

                    if i == 0 and not sus.at[su,"cyclic_state_of_charge"]:
                        previous_state_of_charge = sus.at[su,"state_of_charge_initial"]
                        soc[su,sn][2] -= ((1-sus.at[su,"standing_loss"])**elapsed_hours
                                  * previous_state_of_charge)
                    else:
                        previous_state_of_charge = model.extra_state_of_charge[su,all_snapshots[i-1]]
                        soc[su,sn][0].append(((1-sus.at[su,"standing_loss"])**elapsed_hours,
                                      previous_state_of_charge))

                    state_of_charge = model.extra_state_of_charge[su,sn]

                    soc[su,sn][0].append((-1,state_of_charge))

                    soc[su,sn][0].append((sus.at[su,"efficiency_store"]
                                  * elapsed_hours,model.storage_p_store[su,design_sn]))
                    soc[su,sn][0].append((-(1/sus.at[su,"efficiency_dispatch"]) * elapsed_hours,
                                  model.storage_p_dispatch[su,design_sn]))
                    soc[su,sn][2] -= inflow.at[design_sn,su] * elapsed_hours

                    if su in network.storage_units_t.inflow.keys() and inflow.at[design_sn,su] > 0:
                        storage_p_spill = model.storage_p_spill[su,design_sn]
                        soc[su,sn][0].append((-1.*elapsed_hours,storage_p_spill))

        l_constraint(model,"extra_state_of_charge_constraint",
                     soc, list(network.storage_units.index), all_snapshots)

        model.state_of_charge_constraint.deactivate()

        ### build linecap constraint
        if line_options is not None:
            model.line_volume_limit = pypsa.opt.Constraint(expr=sum(model.link_p_nom[link]*network.links.at[link,"length"]
                for link in network.links.index) <= line_options['line_volume_limit_factor']*line_options['line_volume_limit_max'])

    return(extra_functionality)


### function to modify the soc constraint of pypsa.network.lopf()
### in order to decouple design periods
### possibly also includes an overall line capacity cap constraint
### returns a function to be passed as extra_functionality to pypsa.network.lopf()

def decouple_design_periods(periods, line_options=None):
    
    def extra_functionality(network, snapshots):
        
        model = network.model
        
        ### build soc constraint
        sus = network.storage_units
        
        inflow = network.storage_units_t.inflow
        
        state_of_charge_set = get_switchable_as_dense(network, 'StorageUnit', 'state_of_charge_set', snapshots)
        
        soc = {}

        for period in np.unique(periods):
            start = snapshots[periods==period][0]
            end = snapshots[periods==period][np.sum(periods==period)-1]
            
            snapshots_period = snapshots[periods==period]
    
            for su in sus.index:
            
                for i,sn in enumerate(snapshots_period):
                    
                    soc[su,sn] =  [[],"==",0.]

                    elapsed_hours = 1
                    
                    state_of_charge = model.state_of_charge[su,sn]
                    soc[su,sn][0].append((-1,state_of_charge))
                    
                    if i == 0:
                        soc[su,sn][0].append((1, model.state_of_charge[su,end]))
                    else:
                        previous_state_of_charge = model.state_of_charge[su,snapshots_period[i-1]]

                        soc[su,sn][0].append(((1-sus.at[su,"standing_loss"])**elapsed_hours,
                                      previous_state_of_charge))

                        soc[su,sn][0].append((sus.at[su,"efficiency_store"]
                                  * elapsed_hours,model.storage_p_store[su,sn]))
                        soc[su,sn][0].append((-(1/sus.at[su,"efficiency_dispatch"]) * elapsed_hours,
                                  model.storage_p_dispatch[su,sn]))
                        if su in inflow.keys() and inflow.at[sn,su] > 0:
                            soc[su,sn][2] -= inflow.at[sn,su] * elapsed_hours
                            storage_p_spill = model.storage_p_spill[su,sn]
                            soc[su,sn][0].append((-1.*elapsed_hours,storage_p_spill))
                        

        l_constraint(model,"period_state_of_charge_constraint",
                 soc,list(sus.index), snapshots)
                
        ### build linecap constraint
        if line_options is not None:
            model.line_volume_limit = pypsa.opt.Constraint(expr=sum(model.link_p_nom[link]*network.links.at[link,"length"] 
                for link in network.links.index) <= line_options['line_volume_limit_factor']*line_options['line_volume_limit_max'])

        model.state_of_charge_constraint.deactivate()
        
    return(extra_functionality)
    
 
def inter_intra_soc(periods,day_map,line_options=None):
    """
    Extra functionality following Kotzur et al.
    The idea here calculate soc for every typical day independently (intra_soc)
    and then "copy" them as blocks though the year (inter_soc)
    
    Parameters
    ----------
    periods: list or index slice
        From 0 to the number of typical periods (typically days or weeks),
        each repeated the number of snapshots in each periods (typically 24 or 168)
    
    day_map: numpy array
        The array of the assigned typical periods in order
        (that is numbers from 0 to the number of typical periods in list of lenght 365)
        fromt the tsam package, its the output of from "clusterOrder"
        WARNING! Need to be reordered accordingly if we reorder!
    
    
    Returns
    -------
    None
    """
    
    def extra_functionality(network, snapshots):

        model = network.model

        ### This is the list of storage units and inflow
        sus = network.storage_units
        inflow = network.storage_units_t.inflow

        #Here I define a substitute for model.state_of_charge so that it can be "negative" Real
        # I don't know how to change it in PyPSA, so I will just make the old PyPSA model.state_of_charge useless at the end
        #This represents SOC^intra, just for the typical periods and starting at 0 every time
        #Notice SOC^intra is the state of charge AT THE BEGGINING of the hour
        model.intra_state_of_charge = Var(list(sus.index), snapshots,
                            domain=Reals, bounds=(None,None))
        
        
        ### Here we add the SOC to get it in PyPSa although it is never used as a variable
        #updatesoc = {(su,sn) : [[(-1,model.state_of_charge[su,sn])],"==",0.]
        #         for su in sus.index for sn in snapshots}   
        
        #inicialize the dictionary that will contain the restriction as a pyomo constraint, but really using "l_constraint" from PyPSA
        #for SOC^intra
        soc_intra = {}        
        #we itereate for every typical period (day)
        for period in np.unique(periods):           
            #pick the snapshots of the typical period we are dealing with (e.g. 24 hours)
            snapshots_period = snapshots[periods==period]
            #iterate for every storge unit
            for su in sus.index:
                #iterate for every snapshot in the typical period (hour)
                for i,sn in enumerate(snapshots_period):
                    #inicialize the constraints for soc^intra_su
                    soc_intra[su,sn] =  [[],"==",0.]
                    #for he intra period the elapsed time is just one hour
                    elapsed_hours = 1
                    #put -SOC^intra_s,k,i at the left of the equation (with negative sign!)
                    state_of_charge = model.intra_state_of_charge[su,sn]
                    soc_intra[su,sn][0].append((-1,state_of_charge))
                    
                    # We want the previous_state_of_charge to be zero
                    #for the first snapshot of each typical period (i.e SOC^intra_s,k,1 = 0)
                    if i > 0:
                        #add SOC^intra_s,k,i-1(1-efficiency_loss) at the left of the equation
                        previous_state_of_charge = model.intra_state_of_charge[su,snapshots_period[i-1]]
                        soc_intra[su,sn][0].append(((1-sus.at[su,"standing_loss"])**elapsed_hours,
                                      previous_state_of_charge))
                    #add efficiency_charge*E^char_s,k,i-1 at the left of the equation
                    soc_intra[su,sn][0].append((sus.at[su,"efficiency_store"]
                              * elapsed_hours,model.storage_p_store[su,sn]))
                    #add -(1/efficiency_dis)*E^dis_s,k,i-1 at the left of the equation
                    soc_intra[su,sn][0].append((-(1/sus.at[su,"efficiency_dispatch"]) * elapsed_hours,
                              model.storage_p_dispatch[su,sn]))
                    #add the spill too
                    if su in inflow.keys() and inflow.at[sn,su] > 0:
                        soc_intra[su,sn][2] -= inflow.at[sn,su] * elapsed_hours
                        storage_p_spill = model.storage_p_spill[su,sn]
                        soc_intra[su,sn][0].append((-1.*elapsed_hours,storage_p_spill))
                        
                    ####Check:
                    #if su == sus.index.min():
                    #    print(i, sn)
        #This is the block of constraints concerning SOC^intra
        l_constraint(model,"intra_state_of_charge_constraint",
                 soc_intra,list(sus.index), snapshots)

        #We want to distinguish extendable and non-extendable storage units!
        ext_sus_i = sus.index[sus.p_nom_extendable]
        fix_sus_i = sus.index[~ sus.p_nom_extendable]

        inflow = get_switchable_as_dense(network, 'StorageUnit', 'inflow', snapshots)
        #spill is not implemented here!!!
        #spill_sus_i = sus.index[inflow.max()>0] #skip storage units without any inflow
        #inflow_gt0_b = inflow>0
        #spill_bounds = {(su,sn) : (0,inflow.at[sn,su])
        #            for su in spill_sus_i
        #            for sn in snapshots
        #            if inflow_gt0_b.at[sn,su]}
        #spill_index = spill_bounds.keys()
      
        #This is just the list [0,...,364]
        all_days = list(range(len(day_map)))
        
        ###Define new state of charge variables SOC^inter
        #Hence getting variable for each day of the year
        #Notice that this is the state of charge AT THE END OF THE PERIOD (a bit confusing with SOC^intra)
        model.inter_state_of_charge = Var(list(network.storage_units.index), all_days,
                                    domain=NonNegativeReals, bounds=(0,None))
        
        #Here we take the elapsed hours inside a period (i.e. 24) as "elapsed_hors_intra"
        #but we keep the old elapsed_hours since we need it as well
        elapsed_hours_intra = np.count_nonzero(periods==0)    
        
        #for the extendable ones we have the equation SOC^inter <= hour_max*p_nom_opt (where p_nom_opt is a variable defined in PyPSA with the name model.storage_p_nom)
        #I know that is not exactly right, it just assumes that "inside" the day the SOC would not be too wild
        #It is consistent with the fact that sorage units are planed with a security margin avobe the expected max SOC.        
        upper = {(su,sn,sh) : [[(-sus.at[su,"max_hours"],model.storage_p_nom[su])],"<=",0.]
                 for su in ext_sus_i for sn in all_days for sh in list(range(elapsed_hours_intra))}
        #for the non-extendable ones we just get an upper bound SOC^inter<= hour_max*p_nom ="max SOC" (p_no  is just a number)
        upper.update({(su,sn,sh) : [[],"<=", sus.at[su,"max_hours"]*sus.at[su,"p_nom"]]
                  for su in fix_sus_i for sn in all_days for sh in list(range(elapsed_hours_intra))})
        #we also need a lower bound sinde a storage unit can not have a negative SOC
        lower = {(su,sn,sh) : [[],">=",0.]
                 for su in sus.index for sn in all_days for sh in list(range(elapsed_hours_intra))}
    
        ### inicialize the dict for the constraints concerning SOC^inter
        soc_inter = {}
        #iterate for every storage unit
        for su in sus.index:
                #we do that for all days in the year
                for i,sn in enumerate(all_days):
                    #inicialize the constraint for SOC^inter_s,i
                    soc_inter[su,sn] =  [[],"==",0.]                    
                                     
                    if i == 0 and not sus.at[su,"cyclic_state_of_charge"]:
                        #If non-cyclic put the constant coeff. -(1-efficiency_loss)*initial_set_SOC the the right part of the equation
                        previous_state_of_charge = sus.at[su,"state_of_charge_initial"]
                        soc_inter[su,sn][2] -= ((1-sus.at[su,"standing_loss"])**elapsed_hours_intra
                                  * previous_state_of_charge)

                    else:
                        #In the cyclic case we want SOC^inter_1=SOC^inter_s,N+1 
                        #so we start by adding SOC^inter_s,i-1(1-efficiency_loss)^24 at the right part of the equation (notice Python cyclic notation!)
                        previous_state_of_charge = model.inter_state_of_charge[su,all_days[i-1]]
                        soc_inter[su,sn][0].append(((1-sus.at[su,"standing_loss"])**elapsed_hours_intra,
                                      previous_state_of_charge))
                    #Here we update the upper and lower bounds
                    for sh in range(elapsed_hours_intra):
                        #inter_state_of_charge=model.inter_state_of_charge[su,sn]
                        upper[su,sn,sh][0].append(((1-sus.at[su,"standing_loss"])**sh,previous_state_of_charge))
                        lower[su,sn,sh][0].append(((1-sus.at[su,"standing_loss"])**sh,previous_state_of_charge))
                        ####Add here the actual soc for pypsa
                        #auxvar = (snapshots[periods == day_map[sn]][sh]-pd.Timestamp('2011, 1, 1')).days
                        #if su == sus.index.min() and sh == 0:                            
                        #    print(auxvar, sn, snapshots[periods == day_map[sn]][0])
                        #if sn == auxvar:
                        #    updatesoc[su,snapshots[periods==day_map[sn]][sh]][0].append(((1-sus.at[su,"standing_loss"])**sh,previous_state_of_charge))
              
                    #we need to compute the corresponding date for day_map, i.e. f(i),N (the last hour of the corresponding typical day)
                    design_sn=snapshots[periods==day_map[sn]][np.sum(periods==day_map[0])-1]
                    #put -SOC^inter_s,i at the left of the equation (with negative sign!)
                    state_of_charge = model.inter_state_of_charge[su,sn]
                    soc_inter[su,sn][0].append((-1,state_of_charge))   
                    #put SOC^intra_s,f(i),N
                    last_intra_state_of_charge = model.intra_state_of_charge[su,design_sn]
                    soc_inter[su,sn][0].append((1,last_intra_state_of_charge))
                    '''
                    #put (1-efficiency_loss)*SOC^intra_s,f(i),N
                    previous_intra_state_of_charge = model.intra_state_of_charge[su,design_sn]
                    soc_inter[su,sn][0].append(((1-sus.at[su,"standing_loss"])**elapsed_hours,
                                  previous_intra_state_of_charge))
                    #put efficiency_char*E^char_s,f(i),N
                    soc_inter[su,sn][0].append((sus.at[su,"efficiency_store"]
                                  * elapsed_hours,model.storage_p_store[su,design_sn]))
                    #put -(1/eficiency_dis)*E^dis_s,f(i),N
                    soc_inter[su,sn][0].append((-(1/sus.at[su,"efficiency_dispatch"]) * elapsed_hours,
                                  model.storage_p_dispatch[su,design_sn]))
                    #add the inflow as well
                    soc_inter[su,sn][2] -= inflow.at[design_sn,su] * elapsed_hours
                    if su in network.storage_units_t.inflow.keys() and inflow.at[design_sn,su] > 0:
                        storage_p_spill = model.storage_p_spill[su,design_sn]
                        soc_inter[su,sn][0].append((-1.*elapsed_hours,storage_p_spill))
                    '''    
                    #Here adapt the upper bound inside that inter period    
                    int_periods=snapshots[periods==day_map[sn]]
                    #Notice that in the non-cyclic case, we would skip the SOC at the end of the last day!
                    #We would need an extra variable!
                    
                    
                    for sh in range(1,elapsed_hours_intra):
                        #inter_state_of_charge=model.inter_state_of_charge[su,sn]
                        #upper[su,sn,sh][0].append(((1-sus.at[su,"standing_loss"])**sh,inter_state_of_charge))
                        #lower[su,sn,sh][0].append(((1-sus.at[su,"standing_loss"])**sh,inter_state_of_charge))
                        intra_state_of_charge=model.intra_state_of_charge[su,int_periods[sh-1]]
                        upper[su,sn,sh][0].append((1,intra_state_of_charge))
                        lower[su,sn,sh][0].append((1,intra_state_of_charge))
                        ### here update soc fro PyPSA
                        #print((snapshots[periods == day_map[sn]][sh]-pd.Timestamp('2011, 1, 1')).days)
                        #auxvar = (snapshots[periods == day_map[sn]][sh]-pd.Timestamp('2011, 1, 1')).days
                        #if sn == auxvar:
                        #    updatesoc[su,snapshots[periods==day_map[sn]][sh]][0].append((1,intra_state_of_charge))
                        #    #check
                        #    if su == sus.index.min() and sh == 1:
                        #        print(snapshots[periods == day_map[sn]][sh])
                        
        #Make the SOC^inter constrains
        l_constraint(model,"inter_state_of_charge_constraint",
                     soc_inter, list(network.storage_units.index), all_days)
        
 
        #Here we define the upper bound
        l_constraint(model, "inter_state_of_charge_upper", upper,
             list(network.storage_units.index), all_days, list(range(elapsed_hours_intra)))
        #Here we define the lower bound
        l_constraint(model, "inter_state_of_charge_lower", lower,
             list(network.storage_units.index), all_days, list(range(elapsed_hours_intra)))
        #Here update soc:
        #l_constraint(model, "update_state_of_charge", updatesoc,
        #     list(network.storage_units.index), snapshots )
        
        #Finally deactivate the constrains made by PyPSA
        model.state_of_charge_constraint.deactivate()
        
        ### build linecap constraint
        if line_options is not None:
            model.line_volume_limit = pypsa.opt.Constraint(expr=sum(model.link_p_nom[link]*network.links.at[link,"length"]
                for link in network.links.index) <= line_options['line_volume_limit_factor']*line_options['line_volume_limit_max'])

    return(extra_functionality)


def inter_intra2_soc(periods,day_map,line_options=None):
    """
    Extra functionality following Kotzur et al.
    The idea here calculate soc for every typical day independently (intra_soc)
    and then "copy" them as blocks though the year (inter_soc)
    This is the version given in Appendix B
    
    Parameters
    ----------
    periods: list or index slice
        From 0 to the number of typical periods (typically days or weeks),
        each repeated the number of snapshots in each periods (typically 24 or 168)
    
    day_map: numpy array
        The array of the assigned typical periods in order
        (that is numbers from 0 to the number of typical periods in list of lenght 365)
        fromt the tsam package, its the output of from "clusterOrder"
    
    
    Returns
    -------
    None
    """
    
    def extra_functionality(network, snapshots):

        model = network.model

        ### This is the list of storage units and inflow
        sus = network.storage_units
        inflow = network.storage_units_t.inflow

        #Here I define a substitute for model.state_of_charge so that it can be "negative" Real
        # I don't know how to change it in PyPSA, so I will just make the old PyPSA model.state_of_charge useless at the end
        #This represents SOC^intra, just for the typical periods and starting at 0 every time
        #Notice SOC^intra is the state of charge AT THE BEGGINING of the hour
        model.intra_state_of_charge = Var(list(sus.index), snapshots,
                            domain=Reals, bounds=(None,None))
        
        #We want to distinguish extendable and non-extendable storage units!
        ext_sus_i = sus.index[sus.p_nom_extendable]
        fix_sus_i = sus.index[~ sus.p_nom_extendable]
        
        inflow = get_switchable_as_dense(network, 'StorageUnit', 'inflow', snapshots)
        #spill is not implemented here!!!
        #spill_sus_i = sus.index[inflow.max()>0] #skip storage units without any inflow
        #inflow_gt0_b = inflow>0
        #spill_bounds = {(su,sn) : (0,inflow.at[sn,su])
        #            for su in spill_sus_i
        #            for sn in snapshots
        #            if inflow_gt0_b.at[sn,su]}
        #spill_index = spill_bounds.keys()
      
        #This is just the list [0,...,364]
        all_days = list(range(len(day_map)))
        
        #Here we take the elapsed hours inside a period (i.e. 24) as "elapsed_hors_intra"
        #but we keep the old elapsed_hours since we need it as well
        elapsed_hours_intra = np.count_nonzero(periods==0)    
        
        #for the extendable ones we have the equation SOC^inter <= hour_max*p_nom_opt (where p_nom_opt is a variable defined in PyPSA with the name model.storage_p_nom)
        #I know that is not exactly right, it just assumes that "inside" the day the SOC would not be too wild
        #It is consistent with the fact that sorage units are planed with a security margin avobe the expected max SOC.        
        upper = {(su,sn) : [[(-sus.at[su,"max_hours"],model.storage_p_nom[su])],"<=",0.]
                 for su in ext_sus_i for sn in all_days}
        #for the non-extendable ones we just get an upper bound SOC^inter<= hour_max*p_nom ="max SOC" (p_no  is just a number)
        upper.update({(su,sn) : [[],"<=", sus.at[su,"max_hours"]*sus.at[su,"p_nom"]]
                  for su in fix_sus_i for sn in all_days})
        #we also need a lower bound sinde a storage unit can not have a negative SOC
        lower = {(su,sn) : [[],">=",0.]
                 for su in sus.index for sn in all_days}
        #We define two auxiliary variablies like in Kotzur et al. Appendix B:
        #They detect the maximum in each typical period
        model.max_intra_state_of_charge = Var(list(sus.index), np.unique(periods),
                    domain=Reals, bounds=(None,None))
        model.min_intra_state_of_charge = Var(list(sus.index), np.unique(periods),
                    domain=Reals, bounds=(None,None))
        maximum = {(su,period,sh) : [[(-1,model.max_intra_state_of_charge[su,period])],"<=",0.]
                 for su in sus.index for period in np.unique(periods) for sh in list(range(elapsed_hours_intra))}
        minimum = {(su,period,sh) : [[(-1,model.min_intra_state_of_charge[su,period])],">=",0.]
                 for su in sus.index for period in np.unique(periods) for sh in list(range(elapsed_hours_intra))}
        
        #inicialize the dictionary that will contain the restriction as a pyomo constraint, but really using "l_constraint" from PyPSA
        #for SOC^intra
        soc_intra = {}        
        #we itereate for every typical period (day)
        for period in np.unique(periods):           
            #pick the snapshots of the typical period we are dealing with (e.g. 24 hours)
            snapshots_period = snapshots[periods==period]
            #iterate for every storge unit
            for su in sus.index:
                #iterate for every snapshot in the typical period (hour)
                for i,sn in enumerate(snapshots_period):
                    #inicialize the constraints for soc^intra_su
                    soc_intra[su,sn] =  [[],"==",0.]
                    #for he intra period the elapsed time is just one hour
                    elapsed_hours = 1
                    #put -SOC^intra_s,k,i at the left of the equation (with negative sign!)
                    state_of_charge = model.intra_state_of_charge[su,sn]
                    soc_intra[su,sn][0].append((-1,state_of_charge))
                    
                    #Here we impose SOC^intra to be under the maximum
                    maximum[su,period,i][0].append((1,state_of_charge))
                    #Here impose SOC^intra to be above the minimum
                    minimum[su,period,i][0].append((1,state_of_charge))                    
                    # We want the previous_state_of_charge to be zero
                    #for the first snapshot of each typical period (i.e SOC^intra_s,k,1 = 0)
                    if i > 0:
                        #add SOC^intra_s,k,i-1(1-efficiency_loss) at the left of the equation
                        previous_state_of_charge = model.intra_state_of_charge[su,snapshots_period[i-1]]
                        soc_intra[su,sn][0].append(((1-sus.at[su,"standing_loss"])**elapsed_hours,
                                      previous_state_of_charge))
                    #add efficiency_charge*E^char_s,k,i-1 at the left of the equation
                    soc_intra[su,sn][0].append((sus.at[su,"efficiency_store"]
                              * elapsed_hours,model.storage_p_store[su,sn]))
                    #add -(1/efficiency_dis)*E^dis_s,k,i-1 at the left of the equation
                    soc_intra[su,sn][0].append((-(1/sus.at[su,"efficiency_dispatch"]) * elapsed_hours,
                              model.storage_p_dispatch[su,sn]))
                    #add the spill too
                    if su in inflow.keys() and inflow.at[sn,su] > 0:
                        soc_intra[su,sn][2] -= inflow.at[sn,su] * elapsed_hours
                        storage_p_spill = model.storage_p_spill[su,sn]
                        soc_intra[su,sn][0].append((-1.*elapsed_hours,storage_p_spill))
        #This is the block of constraints concerning SOC^intra
        l_constraint(model,"intra_state_of_charge_constraint",
                 soc_intra,list(sus.index), snapshots)

        #This is the block of constraints that define the maximum and the minimum
        l_constraint(model,"max_intra_state_of_charge_constraint",
                 maximum, sus.index, np.unique(periods), list(range(elapsed_hours_intra)))
        l_constraint(model,"min_intra_state_of_charge_constraint",
                 minimum, sus.index, np.unique(periods), list(range(elapsed_hours_intra)))
        


        
        ###Define new state of charge variables SOC^inter
        #Hence getting variable for each day of the year
        #Notice that this is the state of charge AT THE END OF THE PERIOD (a bit confusing with SOC^intra)
        model.inter_state_of_charge = Var(list(network.storage_units.index), all_days,
                                    domain=NonNegativeReals, bounds=(0,None))
        

        

        
        
        ### inicialize the dict for the constraints concerning SOC^inter
        soc_inter = {}
        #iterate for every storage unit
        for su in sus.index:
                #we do that for all days in the year
                for i,sn in enumerate(all_days):
                    #inicialize the constraint for SOC^inter_s,i
                    soc_inter[su,sn] =  [[],"==",0.]                    
             
                  
                    if i == 0 and not sus.at[su,"cyclic_state_of_charge"]:
                        #If non-cyclic put the constant coeff. -(1-efficiency_loss)*initial_set_SOC the the right part of the equation
                        previous_state_of_charge = sus.at[su,"state_of_charge_initial"]
                        soc_inter[su,sn][2] -= ((1-sus.at[su,"standing_loss"])**elapsed_hours_intra
                                  * previous_state_of_charge)
                    else:
                        #In the cyclic case we want SOC^inter_1=SOC^inter_s,N+1 
                        #so we start by adding SOC^inter_s,i-1(1-efficiency_loss)^24 at the right part of the equation (notice Python cyclic notation!)
                        previous_state_of_charge = model.inter_state_of_charge[su,all_days[i-1]]
                        soc_inter[su,sn][0].append(((1-sus.at[su,"standing_loss"])**elapsed_hours_intra,
                                      previous_state_of_charge))
                    
                    #Here we update the upper and lower bounds
                    #inter_state_of_charge=model.inter_state_of_charge[su,sn]
                    upper[su,sn][0].append((1,previous_state_of_charge))
                    lower[su,sn][0].append(((1-sus.at[su,"standing_loss"])**elapsed_hours_intra,previous_state_of_charge))   
                    
                    #we need to compute the corresponding date for day_map, i.e. f(i),N (the last hour of the corresponding typical day)
                    design_sn=snapshots[periods==day_map[sn]][np.sum(periods==day_map[0])-1]
                    #put -SOC^inter_s,i at the left of the equation (with negative sign!)
                    state_of_charge = model.inter_state_of_charge[su,sn]
                    soc_inter[su,sn][0].append((-1,state_of_charge))                    
                    #put SOC^intra_s,f(i),N
                    last_intra_state_of_charge = model.intra_state_of_charge[su,design_sn]
                    soc_inter[su,sn][0].append((1,last_intra_state_of_charge))
                    '''
                    #put (1-efficiency_loss)*SOC^intra_s,f(i),N
                    previous_intra_state_of_charge = model.intra_state_of_charge[su,design_sn]
                    soc_inter[su,sn][0].append(((1-sus.at[su,"standing_loss"])**elapsed_hours,
                                  previous_intra_state_of_charge))
                    #put efficiency_char*E^char_s,f(i),N
                    soc_inter[su,sn][0].append((sus.at[su,"efficiency_store"]
                                  * elapsed_hours,model.storage_p_store[su,design_sn]))
                    #put -(1/eficiency_dis)*E^dis_s,f(i),N
                    soc_inter[su,sn][0].append((-(1/sus.at[su,"efficiency_dispatch"]) * elapsed_hours,
                                  model.storage_p_dispatch[su,design_sn]))
                    #add the inflow as well
                    soc_inter[su,sn][2] -= inflow.at[design_sn,su] * elapsed_hours
                    if su in network.storage_units_t.inflow.keys() and inflow.at[design_sn,su] > 0:
                        storage_p_spill = model.storage_p_spill[su,design_sn]
                        soc_inter[su,sn][0].append((-1.*elapsed_hours,storage_p_spill))
                    '''    
                    #Here adapt the upper bound inside that inter period
                    #Notice that in case min/max is the last value, we are repeating the computation of SOC_s,i
                    #instead of using it directly
                    #we could do it differently including the value 0 in the min/max and excluding the last "sh"
                    max_intra_state_of_charge=model.max_intra_state_of_charge[su,day_map[sn]]
                    upper[su,sn][0].append((1,max_intra_state_of_charge))
                    min_intra_state_of_charge=model.min_intra_state_of_charge[su,day_map[sn]]
                    lower[su,sn][0].append((1,min_intra_state_of_charge))
                    
                     
        #Make the SOC^inter constrains
        l_constraint(model,"inter_state_of_charge_constraint",
                     soc_inter, list(network.storage_units.index), all_days)
        
 
        #Here we define the upper bound
        l_constraint(model, "inter_state_of_charge_upper", upper,
             list(network.storage_units.index), all_days)
        #Here we define the lower bound
        l_constraint(model, "inter_state_of_charge_lower", lower,
             list(network.storage_units.index), all_days)

        
        #Finally deactivate the constrains made by PyPSA
        model.state_of_charge_constraint.deactivate()
        
        ### build linecap constraint
        if line_options is not None:
            model.line_volume_limit = pypsa.opt.Constraint(expr=sum(model.link_p_nom[link]*network.links.at[link,"length"]
                for link in network.links.index) <= line_options['line_volume_limit_factor']*line_options['line_volume_limit_max'])

    return(extra_functionality)
