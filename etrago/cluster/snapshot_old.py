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

__copyright__ = ("Flensburg University of Applied Sciences, "
                 "Europa-Universität Flensburg, "
                 "Centre for Sustainable Energy Systems")
__license__ = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__author__ = "Simon Hilpert"


def snapshot_clustering(network, how='daily', clusters=10):

    network = run(network=network.copy(), n_clusters=clusters,
                  how=how, normed=False)
    return network


def tsam_cluster(timeseries_df, typical_periods=10, how='daily'):
    """
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with timeseries to cluster

    Returns
    -------
    timeseries : pd.DataFrame
        Clustered timeseries
    """

    if how == 'daily':
        hours = 24
    if how == 'weekly':
        hours = 168
    
    '''
    ###########################################################    
    #URI: This adds peaks
    
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
            
    aggregation = tsam.TimeSeriesAggregation(
    timeseries_df,
    noTypicalPeriods=typical_periods,
    rescaleClusterPeriods=False,
    hoursPerPeriod=hours,
    clusterMethod='hierarchical', 
    extremePeriodMethod = 'append', #'None', 'append', 'new_cluster_center', 'replace_cluster_center'
    addPeakMax = [peakloadcol,peakwindcol,peaksolarcol])
    '''
     
    
    ##################################################################
    #URI: This is the original method=hierarchical
    aggregation = tsam.TimeSeriesAggregation(
        timeseries_df,
        noTypicalPeriods=typical_periods,
        rescaleClusterPeriods=False,
        hoursPerPeriod=hours,
        clusterMethod='hierarchical') #averaging, k_means, k_medoids, hierarchical
    
    timeseries = aggregation.createTypicalPeriods()
    #URI: Better take the whole thing
    timeseries_new =aggregation.predictOriginalData()
    cluster_weights = aggregation.clusterPeriodNoOccur

    
    # get the medoids/ the clusterCenterIndices
    clusterCenterIndices = aggregation.clusterCenterIndices
    
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
    '''
    
    ######################################################
    #URI: This is them method=k_means
    aggregation = tsam.TimeSeriesAggregation(
    timeseries_df,
    noTypicalPeriods=typical_periods,
    rescaleClusterPeriods=False,
    hoursPerPeriod=hours,
    clusterMethod='k_means') #averaging, k_means, k_medoids, hierarchical

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
    '''

    ##########################################################
    #URI: his is to get the data as csv
    directory = '/home/raventos/Results/tsamdata/'
    timeseries_df.to_csv(directory + 'rawPeriods' + str(typical_periods) + '.csv')
    timeseries.to_csv(directory + 'typPeriods' + str(typical_periods) + '.csv')
    timeseries_new.to_csv(directory + 'predictedPeriods' + str(typical_periods) + '.csv')
    aggregation.indexMatching().to_csv(directory + 'indexMatching' + str(typical_periods) + '.csv')
    np.savetxt(directory + 'clusterOrder' + str(typical_periods) + '.csv', aggregation.clusterOrder.astype(int), fmt='%i', delimiter=",")
    cCI=pd.DataFrame(clusterCenterIndices,columns=['index'])
    cCI.to_csv(directory + 'clusterCenterIndices' + str(typical_periods) + '.csv')
    noOccur=pd.DataFrame(cluster_weights, index=['noOccur'])
    noOccur.to_csv(directory + 'noOccurrances' + str(typical_periods) + '.csv')
    aggregation.accuracyIndicators().to_csv(directory + 'indicators' + str(typical_periods) + '.csv')
    np.savetxt(directory + 'dates' + str(typical_periods) + '.csv',dates.strftime("%Y-%m-%d %X"), fmt="%s", delimiter=",")


    #URI: Just checking
    #print([peakloadcol,peakwindcol,peaksolarcol])
    #print(clusterCenterIndices)
    #print([peakloadday,peakwindday,peaksolarday])
    #print(dates)
    #print(cluster_weights)
    #print(aggregation.indexMatching())
    return timeseries_new, cluster_weights, dates, hours


def run(network, n_clusters=None, how='daily',
        normed=False):
    """
    """
    # reduce storage costs due to clusters
    network.cluster = True

    # calculate clusters
    tsam_pre, divisor= prepare_pypsa_timeseries(network)
    tsam_ts, cluster_weights, dates, hours = tsam_cluster(
            tsam_pre, typical_periods=n_clusters,
            how=how)
    #URI: It used to be (wrong): how = 'daily'

    update_data_frames(network, tsam_ts, divisor, cluster_weights, dates, hours)

    return network


def prepare_pypsa_timeseries(network, normed=False):
    """
    """

    if normed:
        normed_loads = network.loads_t.p_set / network.loads_t.p_set.max()
        normed_renewables = network.generators_t.p_max_pu

        df = pd.concat([normed_renewables,
                        normed_loads], axis=1)
    else:
        loads = network.loads_t.p_set
        renewables = network.generators_t.p_max_pu #URI:previously .p_set
        #URI: This divisor will simplify the update
        divisor= len(renewables.columns)
        df = pd.concat([renewables, loads], axis=1)

    return df, divisor


def update_data_frames(network, tsam_ts, divisor, cluster_weights, dates, hours):
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
    #URI: If we want it to really work:
    network.snapshots=network.snapshots.sort_values()
    network.snapshot_weightings=network.snapshot_weightings.sort_index()  
    #URI: Need to separate generators from load
    network.generators_t.p_max_pu = tsam_ts.iloc[:,:divisor] #URI:previously .p_set
    network.loads_t.p_set = tsam_ts.iloc[:,divisor:] 
    
    return network


def daily_bounds(network, snapshots):
    """ This will bound the storage level to 0.5 max_level every 24th hour.
    """
    if network.cluster:

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
