import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

import math
import os
import time

from datetime import datetime, timedelta
from itertools import islice
from hdfs3 import HDFileSystem


hdfs = HDFileSystem(
    host='hdfs://iccluster040.iccluster.epfl.ch', port=8020, user='ebouille')
username = 'olam'


def compute_paths(name_starting_station, name_ending_station, weekday, arrival_time, df_stop_times, df_stop_times_trips, df_trips_type, df_calendar, df_station, df_edges_walk, confidence_level=0.8, maximal_length_route="01:30"):
    """
    Computes all the possible paths for a journey.
    The journey's paramaters are given as input.

    Returns a dataframe with all steps from all possible paths.

    Parameters
    ----------
    name_starting_station: string
        Name of source station of the journey
    name_ending_station: string
        Name of the target station of the journey
    weekday: string
        Day of the week on which the journey takes place
    arrival_time: datetime
        Wanted time of arrival for the journey
    df_stop_times: DataFrame
        Trip, time and sequence info for all stops
        [trip_id|arrival_time|departure_time|stop_id|stop_sequence]
    df_stop_times_trips: DataFrame
        Service info and all of df_stop_times info for all stops
    df_trips_type: DataFrame
        Transport type for all trips
        [trip_id|type]
    df_calendar: DataFrame
        Information about availability of service for each day of the week
        [service_id|monday|tuesday|wednesday|thursday|friday|start_date|end_date]
    df_station: DataFrame
        Information about each stop for all stops
        [stop_id|stop_name|stop_lat|stop_long]
    df_edges_walk: DataFrame
        Walking distance and time information between 2 stops for all eligible stops (<500m)
        [stop_id|stop_id2|distance|travel_time]
    df_edges_public_transportation: DataFrame
        Travel time information between 2 stops for all linked stops
        [stop_id|stop_id2|travel_time]
    confidence_level: float (default: 0.8)
        Confidence threshold for results of possible paths

    Returns
    -------
    DataFrame
        Contains information about all steps for all paths possible for the journey.
        [path_id|stop_id|stop_name|stop_lat|stop_long|transport_type|departure_time]
    """

    # Load statistics data
    print('Loading statistics data... \n')
    df_edges_public_transportations, df_train_stats, df_bus_stats, df_tram_stats = get_stats_given_day(
        weekday)

    # Create static network
    print('Creating the static network...\n')
    G = create_network(df_edges_walk, df_edges_public_transportations)

    # Get the spanning tree from G
    print('Get the spanning tree...\n')
    spanning_tree = create_spanning_tree(name_starting_station, name_ending_station, df_stop_times_trips,
                                         df_calendar, df_station, df_edges_walk, weekday, G, arrival_time, maximal_length_route)

    # Get all the paths and the corresponding prob of success from the spanning tree
    print('Computing the probability of each path...\n')
    all_paths, all_probs, all_transport_type = get_paths_data(
        spanning_tree, weekday, df_train_stats, df_bus_stats, df_tram_stats, df_trips_type, name_starting_station, df_station)

    # Construct the final dataframe for interface
    df_visualization = get_summary(
        all_paths, all_probs, all_transport_type, df_station, confidence_level)

    print('Computing completed!')
    return df_visualization


def load_data():
    """
    Load all the pre-processed data

    Returns
    -------
    df_stop_times: DataFrame
        Trip, time and sequence info for all stops
    df_stop_times_trips: DataFrame
        Service info and all of df_stop_times info for all stops
    df_trips_type: DataFrame
        Transport type for all trips
    df_calendar: DataFrame
        Information about availability of service for each day of the week
    df_station: DataFrame
        Information about each stop for all stops
    df_edges_walk: DataFrame
        Walking distance and time information between 2 stops for all eligible stops (<500m)
    df_edges_public_transportation: DataFrame
        Travel time information between 2 stops for all linked stops
    """

    # Stop times
    files = hdfs.glob(
        '/user/{0}/stop_times.parquet/*.parquet'.format(username))
    df_stop_times = pd.DataFrame()
    for file in files:
        with hdfs.open(file) as f:
            df_stop_times = df_stop_times.append(
                pd.read_parquet(f, engine='pyarrow'))

    # Convert the stop_sequence to a numerical object
    df_stop_times['stop_sequence'] = pd.to_numeric(
        df_stop_times['stop_sequence'])

    # Trips and type
    files = hdfs.glob(
        '/user/{0}/trips_type.parquet/*.parquet'.format(username))
    df_trips_type = pd.DataFrame()
    for file in files:
        with hdfs.open(file) as f:
            df_trips_type = df_trips_type.append(
                pd.read_parquet(f, engine='pyarrow'))

    # Get type 'walking' for trip_id = 'walking'
    walking_row = {}

    walking_row['trip_id'] = 'walking'
    walking_row['type'] = 'walking'

    df_trips_type = df_trips_type.append(walking_row, ignore_index=True)

    # Calendar
    files = hdfs.glob('/user/{0}/calendar.parquet/*.parquet'.format(username))
    df_calendar = pd.DataFrame()
    for file in files:
        with hdfs.open(file) as f:
            df_calendar = df_calendar.append(
                pd.read_parquet(f, engine='pyarrow'))

    # Stations
    files = hdfs.glob(
        '/user/{0}/stations_data.parquet/*.parquet'.format(username))
    df_station = pd.DataFrame()
    for file in files:
        with hdfs.open(file) as f:
            df_station = df_station.append(
                pd.read_parquet(f, engine='pyarrow'))

    # Network walk
    files = hdfs.glob(
        '/user/{0}/edges_walk.parquet/*.parquet'.format(username))
    df_edges_walk = pd.DataFrame()
    for file in files:
        with hdfs.open(file) as f:
            df_edges_walk = df_edges_walk.append(
                pd.read_parquet(f, engine='pyarrow'))

    # Trips
    files = hdfs.glob('/user/{0}/trips.parquet/*.parquet'.format(username))
    df_trips = pd.DataFrame()
    for file in files:
        with hdfs.open(file) as f:
            df_trips = df_trips.append(pd.read_parquet(f, engine='pyarrow'))

    # Create stop_times_trips
    df_stop_times_trips = df_trips.merge(df_stop_times[[
                                         'trip_id', 'arrival_time', 'departure_time', 'stop_id', 'stop_sequence']], on='trip_id', how='inner')
    df_stop_times_trips = df_stop_times_trips[[
        'service_id', 'trip_id', 'arrival_time', 'departure_time', 'stop_id', 'stop_sequence']]

    return df_stop_times, df_stop_times_trips, df_trips_type, df_calendar, df_station, df_edges_walk


def get_stats_given_day(weekday):
    """
    Gets the empirical delay distribution for all types of transport.

    Parameters
    ----------
    weekday: string
        Day of the week at which the journey takes place.

    Returns
    ------
    df_train_stats: DataFrame
        Empirical distribution at the given weekday for trains.
    df_bus_stats: DataFrame
        Empirical distribution at the given weekday for buses.
    df_tram_stats: DataFrame
        Empirical distribution at the given weekday for trams.
    """

    files = hdfs.glob(
        '/user/{}/df_edges_public_transportations_{}.parquet/*.parquet'.format(username, weekday))
    df_edges_public_transportations = pd.DataFrame()
    for file in files:
        with hdfs.open(file) as f:
            df_edges_public_transportations = df_edges_public_transportations.append(
                pd.read_parquet(f, engine='pyarrow'))

    # Load trains
    files = hdfs.glob(
        '/user/{}/stats_delays_{}_Zug.parquet/*.parquet'.format(username, weekday))
    df_train_stats = pd.DataFrame()
    for file in files:
        with hdfs.open(file) as f:
            df_train_stats = df_train_stats.append(
                pd.read_parquet(f, engine='pyarrow'))

    # Load buses
    files = hdfs.glob(
        '/user/{}/stats_delays_{}_Bus.parquet/*.parquet'.format(username, weekday))
    df_bus_stats = pd.DataFrame()
    for file in files:
        with hdfs.open(file) as f:
            df_bus_stats = df_bus_stats.append(
                pd.read_parquet(f, engine='pyarrow'))

    # Load trams
    files = hdfs.glob(
        '/user/{}/stats_delays_{}_Tram.parquet/*.parquet'.format(username, weekday))
    df_tram_stats = pd.DataFrame()
    for file in files:
        with hdfs.open(file) as f:
            df_tram_stats = df_tram_stats.append(
                pd.read_parquet(f, engine='pyarrow'))

    df_train_stats['arrival_time'] = pd.to_datetime(
        df_train_stats['arrival_time'], format='%H:%M:%S').dt.time
    df_bus_stats['arrival_time'] = pd.to_datetime(
        df_bus_stats['arrival_time'], format='%H:%M:%S').dt.time
    df_tram_stats['arrival_time'] = pd.to_datetime(
        df_tram_stats['arrival_time'], format='%H:%M:%S').dt.time

    return df_edges_public_transportations, df_train_stats, df_bus_stats, df_tram_stats


def create_network(df_edges_walk, df_edges_public_transportations):
    """
    Creates the static network of public transport.
    Includes walkings trips if reasonnable (<500m)

    Parameters
    ----------
    df_edges_walk: DataFrame
        Walking distance and time information between 2 stops for all eligible stops (<500m)
    df_edges_public_transportation: DataFrame
        Travel time information between 2 stops for all linked stops

    Returns
    -------
    G: nx.Graph
        Static graph of public transport + walking possibilities
    """
    # Process the edges data
    df_edges_all = pd.concat(
        [df_edges_walk[['stop_id', 'stop_id2', 'travel_time']], df_edges_public_transportations])
    df_edges_all = df_edges_all.dropna()
    df_edges_all['travel_time'] = pd.to_numeric(df_edges_all['travel_time'])

    # Create the network
    G = nx.from_pandas_edgelist(
        df_edges_all, 'stop_id', 'stop_id2', ['travel_time'])

    return G


def k_shortest_paths(G, source, target, k):
    """
    Get the k shortest paths from input graph G.

    Parameters
    ----------
    G: nx.Graph
        Static public transport graph
    """

    try:
        shortest_paths = list(
            islice(nx.shortest_simple_paths(G, source, target, 'travel_time'), k))
    except nx.NetworkXNoPath:
        shortest_paths = []

    return shortest_paths


def get_time_interval(time1, time2):
    """
    Get the time interval in seconds betweem time1 and time2

    Parameters
    ----------
    time1: datetime
        First time of the interval
    time2: datetime
        Second time of the interval
    """
    return (time2.hour - time1.hour) * 3600\
        + (time2.minute - time1.minute) * 60\
        + time2.second - time1.second


def stop_name2stop_id(stop_name, df_stations):
    """
    Get the station's id for the given station's name.
    """
    return df_stations[df_stations['stop_name'].values == stop_name]['stop_id'].iloc[0]


def create_spanning_tree(name_starting_station, name_ending_station, df_stop_times_trips, df_calendar, df_stations, df_edges_walk, weekday, G, arrival_time, maximal_length_route="01:30", num_paths=5):
    """
    Creates a spanning tree with destination as origin node.
    This spanning tree looks for all possibilities of trips from destination to source.

    Parameters
    ----------
    name_starting_station: string
        Name of source station of the journey
    name_ending_station: string
        Name of the target station of the journey
    df_stop_times_trips: DataFrame
        Service info, trip, time and sequence info for all stops
    df_calendar: DataFrame
        Information about availability of service for each day of the week
    df_station: DataFrame
        Information about each stop for all stops
    df_edges_walk: DataFrame
        Walking distance and time information between 2 stops for all eligible stops (<500m)
    weekday: string
        Day of the week at which the journey takes place.
    G: nx.Graph
        Static graph of public transport + walking possibilities
    arrival_time: datetime
        Desired time of arrival
    num_paths: int
        Number of paths to consider for the trips.

    Returns
    -------
    spanning_tree: nx.DiGraph
        Spanning tree of all the possible trips for the journey.
    """

    # Initialize the spanning tree
    spanning_tree = nx.DiGraph()

    # Get departure and arrival stop_id
    departure_stop_id = stop_name2stop_id(name_starting_station, df_stations)
    arrival_stop_id = stop_name2stop_id(name_ending_station, df_stations)

    # Check which services are available on the given day
    date_selection = df_calendar[['service_id',
                                  weekday]].loc[df_calendar[weekday] == '1']
    # Filter the trips from these services for the given day
    df_stop_times_filtered = date_selection.merge(
        df_stop_times_trips, on='service_id', how='inner')

    # Add the arrival node with the latest time possible (origin node)
    arrival_node_id = arrival_stop_id + '_' + str(arrival_time.time())
    spanning_tree.add_node(arrival_node_id)

    # Get all the 'num_paths' shortest paths from departure to arrival station
    all_paths = k_shortest_paths(
        G, departure_stop_id, arrival_stop_id, num_paths)

    for i, path in enumerate(all_paths):

        print('Exploring path {}...'.format(i + 1))

        # Check that the path is acyclic
        if len(set(path)) == len(path):

            # Expand the spanning tree if the path is acyclic
            expand_spanning_tree(spanning_tree, path[:-1], arrival_node_id, arrival_stop_id, None, df_stop_times_filtered,
                                 df_edges_walk, arrival_time, cumulate_time=0, maximal_length_route=maximal_length_route)

    return spanning_tree


def expand_spanning_tree(spanning_tree, path, next_node_id, next_stop_id, next_trip_id, df_stop_times_filtered, df_edges_walk, max_time, cumulate_time, maximal_length_route):
    """
    For all the stops of the input path, we compute all the possible options from it to the next stop for all the possible times.

    Parameters
    ----------
    spanning_tree: nx.Graph
        Spanning tree to be completed by the path's exploration
    path: list
        List of nodes ids representing a path in the transport network
    next_node_id: string
        id of the next node (stop at a certain time) in the journey
    next_stop_id: string
        id of the next stop in the journey
    next_trip_id: string
        id of the trip following taken after next_stop_id
    df_stop_times_filtered: DataFrame
        Dataframe containing the stop information of the available services at the day of the journey
    df_edges_walk: DataFrame
        Dataframe containing the walking distance and travel information between all eligible stops
    max_time: datetime
        Time before which we need to arrive at next_stop_id
    """
    # We stop the recursion when there is no stop left
    # or if path takes too much time
    max_time_for_path = maximal_length_route.hour * \
        60 * 60 + maximal_length_route.minute * 60

    if len(path) >= 1 and cumulate_time < max_time_for_path:

        # Get the stop
        stop_id = path[-1]

        # Find all trips that go from stop_id to next_stop_id before max_time
        df_trip_ids = get_trip_id(
            df_stop_times_filtered, df_edges_walk, stop_id, next_stop_id, max_time)

        # add node(s) for each trip
        for i, row in df_trip_ids.iterrows():

            # Get information about the trip_id
            trip_id = row['trip_id']
            departure_time = row['departure_time']
            arrival_time = row['arrival_time']
            travel_time = get_time_interval(departure_time, arrival_time)

            cumulate_time += travel_time

            # Get the unique node_id (stop_id and departure time from this stop)
            node_id = stop_id + '_' + str(departure_time) + '_dep'

            # The case when we do not change transport or we walk twice (between 3 stations)
            if (trip_id == next_trip_id) or (trip_id == 'walking'):

                # Add node_id and link it to next_node_id
                spanning_tree.add_node(
                    node_id, stop_id=stop_id, time=departure_time)
                spanning_tree.add_edge(
                    node_id, next_node_id, trip_id=trip_id, travel_time=travel_time)

            # There is a transfer => we need to consider waiting time
            else:

                # Get waiting time
                waiting_time = get_time_interval(arrival_time, max_time)

                cumulate_time += waiting_time

                # If we are going from one transport to another transport and the waiting time is not at least 2 min
                if trip_id and next_trip_id and waiting_time < 120:
                    continue

                # Split the next_stop_id for adding waiting time edge
                next_node_wait_id = next_stop_id + \
                    '_' + str(arrival_time) + '_arr'

                # Add next_node_wait_id node and link it to next_node_id
                spanning_tree.add_node(
                    next_node_wait_id, stop_id=next_stop_id, time=arrival_time)
                spanning_tree.add_edge(
                    next_node_wait_id, next_node_id, trip_id='waiting', travel_time=waiting_time)

                # Add node_id node and link it to next_node_wait_id
                spanning_tree.add_node(
                    node_id, stop_id=stop_id, time=departure_time)
                spanning_tree.add_edge(
                    node_id, next_node_wait_id, trip_id=trip_id, travel_time=travel_time)

            max_time = datetime.strptime(
                '2020-01-01 ' + str(departure_time), '%Y-%m-%d %H:%M:%S')
            expand_spanning_tree(spanning_tree, path[:-1], node_id, stop_id, trip_id,
                                 df_stop_times_filtered, df_edges_walk, max_time, cumulate_time, maximal_length_route)


def get_trip_id(df_stop_times_filtered, df_edges_walk, stop_id, next_stop_id, max_time, interval=15):
    """
    Given a stop_id and the next_stop_id, find all the trip_id from stop_id that arrive at next_stop_id before max_time
    If there is no public transportation between the two stops but we can walk, then trip_id is None

    Parameters
    ----------
    stop_id: str
        id of the stop from which we come from in the journey
    next_stop_id: str
        id of the stop we are now in the journey
    max_time: datetime.datetime
        Time before which we need to arrive at next_stop_id
    interval: int
        The interval of time given to arrive to next_stop_id.
        i.e. one cannot take more than `interval` time to arrive from stop_id to next_stop_id

        The interval of time in minutes we are considering the trip_id to arrive.
        So a trip has to arrive before max_time and after (max_time - interval).

    Returns
    -------
    df_filtered: pd.DataFrame
        DataFrame containing the possible trips between the 2 stops
        [trip_id|departure_time|arrival_time]
    """

    # Get data for each station
    df1 = df_stop_times_filtered[df_stop_times_filtered['stop_id'].values == stop_id]
    df2 = df_stop_times_filtered[df_stop_times_filtered['stop_id'].values == next_stop_id]

    # (Outer) merge the data of each station on the trip_id
    df_merged = df1.merge(df2, on='trip_id', how='outer')

    # Keep only trips for direct trip from stop_id to next_stop_id
    df_filtered = df_merged[df_merged['stop_sequence_x'].apply(
        lambda x: x + 1) == df_merged['stop_sequence_y']]

    # df_filtered may contain null values for arrival_time and departure_time if it is walking, source or destination
    df_filtered = df_filtered.dropna().rename({'arrival_time_y': 'arrival_time',
                                               'departure_time_x': 'departure_time'}, axis=1)

    # Keep only the time for departure_time and arrival_time
    df_filtered['arrival_time'] = df_filtered['arrival_time'].apply(
        lambda x: x.time())
    df_filtered['departure_time'] = df_filtered['departure_time'].apply(
        lambda x: x.time())

    df_temp = []

    # If there exist possible paths to get to next_stop_id with public transport
    if not df_filtered.empty:
        # Interval threshold for connections search
        interval_threshold = 4 * 60

        # Get larger interval if no data remain
        while len(df_temp) == 0 and interval <= interval_threshold:
            min_time = (max_time - timedelta(minutes=interval)).time()

            df_temp = df_filtered[(df_filtered['arrival_time'] < max_time.time()) &
                                  (df_filtered['arrival_time'] > min_time)]
            interval *= 2

        df_filtered = df_temp.copy()

    # Keep only relevant data and reset index
    df_filtered = df_filtered[[
        'trip_id', 'departure_time', 'arrival_time']].reset_index(drop=True)

    # Check if we can go by foot between the two stations
    df_walk = df_edges_walk[(df_edges_walk['stop_id'] == stop_id) &
                            (df_edges_walk['stop_id2'] == next_stop_id)]

    if not df_walk.empty:

        walk_time = float(df_walk['travel_time'])

        new_entry = {}

        new_entry['trip_id'] = 'walking'
        new_entry['departure_time'] = (
            max_time - timedelta(seconds=walk_time)).time().replace(microsecond=0)
        new_entry['arrival_time'] = max_time.time()

        df_filtered = df_filtered.append(new_entry, ignore_index=True)

    return df_filtered


def get_paths_data(spanning_tree, weekday, df_train_stats, df_bus_stats, df_tram_stats, df_trips_type, name_starting_station, df_station):
    """
    Computes the probability of success for each path starting from a departure node in the spanning tree.

    Parameters
    ---------
    spanning_tree: nx.DiGraph
        Spanning tree containing all possible paths for this journey
    weekday: string
        Day of the week at which the journey takes place.
    df_train_stats: DataFrame
        Empirical distribution at the given weekday for trains.
    df_bus_stats: DataFrame
        Empirical distribution at the given weekday for buses.
    df_tram_stats: DataFrame
        Empirical distribution at the given weekday for trams.
    df_trips_type: DataFrame
        Transport type for all trips
    """

    # Get all the nodes (stop_id at a corresponding time) along all the paths

    all_paths = []
    all_probs = []
    all_transport_type = []

    # For computing the probability of success of a path
    transport_type_list = []
    transport_type_list_for_proba = []
    arrival_stop_id_list = []
    previous_stop_id_list = []
    arrival_time_list = []
    max_delay_list = []

    source_stop_id = stop_name2stop_id(name_starting_station, df_station)

    # Create a generator for all departure nodes and iterate to get prob of delay for each path
    for departure_node in (x for x in spanning_tree.nodes() if spanning_tree.in_degree(x) == 0):

        if departure_node[:7] == source_stop_id:

            actual_path = []
            get_prob_per_path(spanning_tree, weekday, departure_node, None, actual_path, all_paths,
                              transport_type_list_for_proba, arrival_stop_id_list,
                              previous_stop_id_list, arrival_time_list, max_delay_list, all_probs, all_transport_type, df_train_stats, df_bus_stats, df_tram_stats, df_trips_type, transport_type_list)

    return all_paths, all_probs, all_transport_type


def get_prob_per_path(spanning_tree, weekday, actual_node, previous_node, actual_path, all_paths, transport_type_list_for_proba, arrival_stop_id_list, previous_stop_id_list, arrival_time_list, max_delay_list, all_probs, all_transport_type, df_train_stats, df_bus_stats, df_tram_stats, df_trips_type, transport_type_list):
    """
    Recursive fonction that explore the next stop for a path.

    Parameters
    ----------
    spanning_tree: nx.DiGraph
        Spanning tree containing all possible paths for this journey
    weekday: string
        Day of the week at which the journey takes place.
    actual_node: string
        Current node in the iteration
    previous_node: string
        id of previous node
    actual_path: list
        List of the path we are considering
    all_paths: list
        List of all paths
    transport_type_list_for_proba: list
        Cumulated transport types to compute probabilities
    arrival_stop_id_list: list
        Cumulated list of stop ids
    previous_stop_id_list: list
        Cumulated list of previous stop ids
    arrival_time_list: list
        Cumulated list of arrival times
    max_delay_list: list
        Cumulated list of max delay times
    all_probs: list
        Probability for each route, computed at leaf
    all_transport_type: list
        Encountered transport types
    df_train_stats: DataFrame
        Empirical distribution at the given weekday for trains.
    df_bus_stats: DataFrame
        Empirical distribution at the given weekday for buses.
    df_tram_stats: DataFrame
        Empirical distribution at the given weekday for trams.
    df_trips_type: DataFrame
        Transport type for all trips
    transport_type_list: list
        List of all transports types.
    """

    neighbors = [x for x in spanning_tree.neighbors(actual_node)]

    actual_path.append(actual_node)

    # If current node is not a departure node, we get the trip id
    if previous_node is not None:
        edge = spanning_tree.get_edge_data(previous_node, actual_node)
        # Get trip id
        trip_id = edge['trip_id']

        # If we came from a transport, we get the transport_type
        if trip_id != 'waiting':
            transport_type = df_trips_type[df_trips_type['trip_id'].values ==
                                           trip_id]['type'].iloc[0]
            # Get transport type
            transport_type_list.append(transport_type)

    # If current node is an arrival and then wait (i.e. transfer, so risk of delay)
    if actual_node[-3:] == 'arr':

        # Get the previous_stop_id and stop_id of the node
        arrival_stop_id = actual_node[:7]
        previous_stop_id = previous_node[:7]

        # Get the arrival time
        arrival_time = datetime.strptime(actual_node[8:16], '%H:%M:%S')

        # Get the previous trip_id
        edge = spanning_tree.get_edge_data(previous_node, actual_node)
        trip_id = edge['trip_id']

        # Get the transport time
        transport_type = df_trips_type[df_trips_type['trip_id'].values ==
                                       trip_id]['type'].iloc[0]

        transport_type_list_for_proba.append(transport_type)
        arrival_stop_id_list.append(arrival_stop_id)
        previous_stop_id_list.append(previous_stop_id)
        arrival_time_list.append(arrival_time)

        # Get max_delay
        time_format = '%H:%M:%S'
        departure_time = actual_node[8:16]
        arrival_time = previous_node[8:16]
        max_delay = datetime.strptime(
            departure_time, time_format) - datetime.strptime(arrival_time, time_format)

        max_delay_list.append(int(max_delay.seconds / 60))

    # If there are neighbors left, recurse
    if len(neighbors) > 0:
        for neighbor in neighbors:

            # We may have multiple neighbours => copy the actual_path before going to two stops
            actual_path_n = actual_path.copy()
            transport_type_list_for_proba_n = transport_type_list_for_proba.copy()
            transport_type_list_n = transport_type_list.copy()
            arrival_stop_id_list_n = arrival_stop_id_list.copy()
            previous_stop_id_list_n = previous_stop_id_list.copy()
            arrival_time_list_n = arrival_time_list.copy()
            max_delay_list_n = max_delay_list.copy()

            get_prob_per_path(spanning_tree, weekday, neighbor, actual_node, actual_path_n, all_paths,
                              transport_type_list_for_proba_n, arrival_stop_id_list_n,
                              previous_stop_id_list_n, arrival_time_list_n, max_delay_list_n, all_probs, all_transport_type, df_train_stats, df_bus_stats, df_tram_stats, df_trips_type, transport_type_list)

    # There are no neighbors left
    else:
        all_paths.append(actual_path)
        all_probs.append(compute_proba(weekday, transport_type_list_for_proba, arrival_stop_id_list,
                                       previous_stop_id_list, arrival_time_list, max_delay_list,
                                       df_train_stats, df_bus_stats, df_tram_stats))
        all_transport_type.append(transport_type_list)


def compute_proba(weekday, transport_type_list, arrival_stop_id_list, previous_stop_id_list, arrival_time_list, max_delay_list, df_train_stats, df_bus_stats, df_tram_stats):
    """
    Computes the cumulated probability of success.
    """

    proba = 1
    for i in range(len(transport_type_list)):

        transport_type = transport_type_list[i]
        stop_id = arrival_stop_id_list[i]
        previous_stop_id = previous_stop_id_list[i]
        arrival_time = arrival_time_list[i]
        max_delay = max_delay_list[i]

        if transport_type == "zug":
            all_delays = get_larger_interval(
                df_train_stats, arrival_time, stop_id)

        elif transport_type == "bus":
            all_delays = get_larger_interval(
                df_bus_stats, arrival_time, stop_id)

        elif transport_type == "tram":
            all_delays = get_larger_interval(
                df_tram_stats, arrival_time, stop_id)

        else:
            raise("This transport doesn't exist in Zurich: expected Zug, Bus or Tram, got {}.".format(
                transport_type))

        proba *= compute_single_proba(all_delays, max_delay)

    return round(proba, 3)


def get_larger_interval(df_transport_type, arrival_time, stop_id, interval=1, threshold=10):
    """
    Tunes the interval to get a sufficient number of historical delays to compute relevant probabilities.
    """

    df_temp = df_transport_type[(df_transport_type["arrival_time"].values == arrival_time.time()) &
                                (df_transport_type["stop_id"].values == stop_id)]

    # Get all the arrival delays for that station at the arrival time
    all_delays = []
    df_temp['delay_list'].apply(lambda x: all_delays.extend(x))

    while len(all_delays) < threshold:

        interval *= 8

        max_time = (arrival_time + timedelta(minutes=interval)).time()
        min_time = (arrival_time - timedelta(minutes=interval)).time()

        # Keep only trip_id that arrive at next_stop_id before max_time
        df_temp = df_transport_type[(df_transport_type['arrival_time'] < max_time) &
                                    (df_transport_type['arrival_time'] > min_time)]

        if interval < 720:
            df_temp = df_temp[df_temp["stop_id"].values == stop_id]

        # Get all the arrival delays for that station at the arrival time
        all_delays = []
        df_temp['delay_list'].apply(lambda x: all_delays.extend(x))

    return all_delays


def compute_single_proba(delay_list, max_delay):
    """
    Computes the actual probability
    """
    delay_list = np.bincount(np.clip(delay_list, a_min=0, a_max=30))
    num_delays = np.sum(delay_list)
    proba = delay_list[:max_delay + 1].sum()

    return proba / num_delays


def get_summary(all_paths, all_probs, all_transport_type, df_station, confidence_threshold):
    """
    Prepares a dataframe with all the results as needed by the visualization.
    """

    # Mapping from German to English transport name
    transport_type_german_eng_mapping = {
        'zug': 'train', 'tram': 'tram', 'bus': 'bus', 'walking': 'walking'}
    df_visualization = []

    for idx, path in enumerate(all_paths):

        if len(path) > 1:
            counter_edge = 0
            for stop_idx, stop in enumerate(path):

                if stop[-3:] == 'dep':
                    stop_id = stop[:7]
                    stop_infos = df_station.loc[df_station['stop_id'].values == stop_id]

                    # Source node
                    if stop_idx == 0:
                        df_visualization.append([idx, stop_id, stop_infos['stop_name'].values[0], stop_infos['stop_lat'].values[0],
                                                 stop_infos['stop_lon'].values[0], transport_type_german_eng_mapping[
                                                     all_transport_type[idx][counter_edge]],
                                                 stop[8:16], 'source', all_probs[idx]])
                    else:
                        df_visualization.append([idx, stop_id, stop_infos['stop_name'].values[0], stop_infos['stop_lat'].values[0],
                                                 stop_infos['stop_lon'].values[0], transport_type_german_eng_mapping[
                                                     all_transport_type[idx][counter_edge]],
                                                 stop[8:16], all_paths[idx][stop_idx - 1][8:16], all_probs[idx]])
                    counter_edge += 1

                # Target node
                elif (stop[-3:] != 'dep') & (stop[-3:] != 'arr'):
                    stop_id = stop[:7]
                    stop_infos = df_station.loc[df_station['stop_id'].values == stop_id]

                    df_visualization.append([idx, stop_id, stop_infos['stop_name'].values[0], stop_infos['stop_lat'].values[0],
                                             stop_infos['stop_lon'].values[0], transport_type_german_eng_mapping[
                                                 all_transport_type[idx][counter_edge - 1]],
                                             'target', all_paths[idx][stop_idx - 1][8:16], all_probs[idx]])

    df_visualization = pd.DataFrame(df_visualization, columns=["path_id", "stop_id", "stop_name",
                                                               "stop_lat", "stop_long", "transport_type",
                                                               "departure_time", "arrival_time",
                                                               "path_probability"])

    if len(df_visualization) > 0:
        # Keep only paths where the probability path is greater than confidence_threshold
        df_visualization = df_visualization[df_visualization['path_probability'].values >=
                                            confidence_threshold]

        # Select only rows corresponding to the source node in order to sort according to departure time
        df_sorted_paths = df_visualization[[
            'path_id', 'departure_time', 'arrival_time', 'path_probability']]
        df_sorted_paths = df_sorted_paths[df_sorted_paths['arrival_time'].values == 'source'][[
            'path_id', 'departure_time', 'path_probability']]
        df_sorted_paths = df_sorted_paths.sort_values(
            by=['departure_time', 'path_probability'], ascending=[False, False])[['path_id']]

        # Merge the sorted path_id dataframe with df_visualization which will result in sorted paths
        df_visualization = df_sorted_paths.merge(
            df_visualization.set_index('path_id'), on='path_id', how='left')

    return df_visualization