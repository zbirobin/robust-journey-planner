import ipywidgets as widgets
import plotly.graph_objects as go
from datetime import datetime
import pandas as pd
import numpy as np
import pickle

import plotly.graph_objects as go

from helpers_algo import compute_paths, load_data


class Displayer:
    """
    This class encompasses the whole Planner and its display.
    Journey planning is done under-the-hood by this class.
    """

    def __init__(self):

        # Widgets
        self.starting_station_widget = None
        self.ending_station_widget = None
        self.weekday_widget = None
        self.arrival_time_widget = None
        self.confidence_widget = None
        self.num_path_on_map_widget = None
        self.length_route_widget = None

        self.station_list = load_pickle('../data/list_stop_names.pickle')

        # Timetable data
        self.df_stop_times = None
        self.df_stop_times_trips = None
        self.df_trips_type = None
        self.df_calendar = None
        self.df_station = None
        self.df_edges_walk = None

        # Result with all the paths
        self.paths_dataframe = None

    def load_tables(self):
        """Load the precomputed HDFS tables necessary for the planner."""

        print("Loading the data... (~ 2 minutes maximum)")
        df_stop_times, df_stop_times_trips, df_trips_type, df_calendar, df_station, df_edges_walk = load_data()

        self.df_stop_times = df_stop_times.copy()
        self.df_stop_times_trips = df_stop_times_trips.copy()
        self.df_trips_type = df_trips_type.copy()
        self.df_calendar = df_calendar.copy()
        self.df_station = df_station.copy()
        self.df_edges_walk = df_edges_walk.copy()

    def display_widgets(self):
        """
        Prepares and displays the widgets for the journey planner interface.
        Default values are arbitrary.
        """

        # Select FROM station
        self.starting_station_widget = widgets.Combobox(
            options=self.station_list.tolist(),
            value='Zürich HB',
            description='',
            disabled=False,
            ensure_option=True
        )

        # Select TO station
        self.ending_station_widget = widgets.Combobox(
            options=self.station_list.tolist(),
            value='Zürich Flughafen, Bahnhof',
            description='',
            disabled=False,
            ensure_option=True
        )

        # Select day of week during which the trip is done
        self.weekday_widget = widgets.Dropdown(
            options=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
            value='Monday',
            description='',
            disabled=False,
        )

        # Select wanted time of arrival
        start_date = datetime(2019, 5, 13, 4)
        end_date = datetime(2019, 5, 13, 23)
        dates = pd.date_range(start_date, end_date, freq='Min')
        self.arrival_time_widget = widgets.SelectionSlider(
            options=[(date.strftime('%H:%M'), date) for date in dates],
            description='',
            index=315,
            orientation='horizontal',
            layout={'width': '320px'}
        )

        # Select threshold for minimum confidence level of arriving on thime
        self.confidence_widget = widgets.BoundedFloatText(
            value=80,
            min=0,
            max=100,
            step=0.5,
            description='',
            disabled=False
        )

        # Control the number of trips to be displayed on the map
        self.num_path_on_map_widget = widgets.BoundedFloatText(
            value=5,
            min=1,
            max=100,
            step=1,
            description='',
            disabled=False
        )

        # Select maximum length of a path
        start_date = datetime(2019, 5, 13, 0)
        end_date = datetime(2019, 5, 13, 4)
        dates = pd.date_range(start_date, end_date, freq='Min')
        self.length_route_widget = widgets.SelectionSlider(
            options=[(date.strftime('%H:%M'), date) for date in dates],
            description='',
            index=120,
            orientation='horizontal',
            layout={'width': '320px'}
        )

        print("\nPlease enter the following values")
        print("\nFrom station:")
        display(self.starting_station_widget)
        print("\nTo station:")
        display(self.ending_station_widget)
        print("\nOn:")
        display(self.weekday_widget)
        print("\nWith arrival time:")
        display(self.arrival_time_widget)
        print("\nWith confidence level Q (%):")
        display(self.confidence_widget)
        print("\nNumber of paths displayed on the map:")
        display(self.num_path_on_map_widget)
        print("\nMaximal length time of the route:")
        display(self.length_route_widget)

    def find_and_display_paths(self):
        """
        Calls the journey planner algorithm with the query parameters chosen by the user.
        Displays the result on a map and prints the instructions for the best one.
        """
        self.paths_dataframe = compute_paths(
            self.starting_station_widget.value,
            self.ending_station_widget.value,
            self.weekday_widget.value.lower(),
            self.arrival_time_widget.value,
            self.df_stop_times,
            self.df_stop_times_trips,
            self.df_trips_type,
            self.df_calendar,
            self.df_station,
            self.df_edges_walk,
            self.confidence_widget.value / 100,
            self.length_route_widget.value
        )
        self.display_map()
        self.print_best_path()

    def display_map(self):
        """
        Displays a Plotly map showing all the routes retourned by the journey planner algorithm.
        """
        paths_dataframe = self.paths_dataframe.copy()

        # Get all unique paths identifiers
        list_path_id = list(paths_dataframe.path_id.unique())
        num_path = len(list_path_id)

        # Journey planner found no routes
        if num_path == 0:
            print("No route was found...")

        # Journey planner found 1+ routes
        else:

            path = paths_dataframe[paths_dataframe['path_id']
                                   == list_path_id[0]]
            departure_time = path.departure_time.values[0]

            # ===========================================================================
            def add_text(texts, row):
                """
                Computes the captions for the journey's steps.
                Captions are different depending on the nature of the step in the journey.
                """
                # Target reached, text displays arrival time.
                if row['departure_time'] == "target":
                    texts.append("{}, arr: {}".format(
                        row['stop_name'], row['arrival_time'][:5]))

                # Source left, the journey just started
                elif row['arrival_time'] == "source":

                    # First step is walking to another station
                    if row['transport_type'] == "walking":
                        texts.append("{}, dep: {}, {}".format(
                            row['stop_name'], row['departure_time'][:5], row['transport_type']))
                    # First step is taking a public transport
                    else:
                        texts.append("{}, dep: {}, {}".format(
                            row['stop_name'], row['departure_time'][:5], row['transport_type']))

                # Normal step during the journey with change of transport type
                elif row['arrival_time']:
                    # We exit transport to walk
                    if row['transport_type'] == "walking":
                        texts.append("{}, arr: {}, {}".format(
                            row['stop_name'], row['arrival_time'][:5], row['transport_type']))
                    # We change our type of transport (e.g. bus)
                    else:
                        texts.append("{}, arr: {}, dep: {}, {}".format(
                            row['stop_name'], row['arrival_time'][:5], row['departure_time'][:5], row['transport_type']))

                # Normal step during the journey without change of transport type
                else:
                    # We stay in the transport
                    texts.append("{}".format(row['stop_name']))
            # ===========================================================================

            print("")

            texts = []
            # Create captions for all steps of all journeys
            for j, row in path.iterrows():
                add_text(texts, row)

            # Create map with path from the first journey
            fig = go.Figure(go.Scattermapbox(
                mode="markers+lines",
                lon=path['stop_long'],
                lat=path['stop_lat'],
                text=texts,
                marker={'size': 10},
                name="Route 1, {}, {:.2f}%".format(departure_time, path['path_probability'].mean() * 100)))

            # Add paths as markers and lines on the map
            for i in range(1, np.min([num_path, int(self.num_path_on_map_widget.value)])):
                # Get all steps for current path
                path = paths_dataframe[paths_dataframe['path_id']
                                       == list_path_id[i]]

                departure_time = path.departure_time.values[0]

                # Compute captions for current path
                texts = []
                for j, row in path.iterrows():
                    add_text(texts, row)

                # Add current path on map
                fig.add_trace(go.Scattermapbox(
                    mode="markers+lines",
                    lon=path['stop_long'],
                    lat=path['stop_lat'],
                    text=texts,
                    marker={'size': 10},
                    name="Route {}, {}, {:.2f}%".format(i + 1, departure_time, path['path_probability'].mean() * 100)))

            fig.update_layout(
                margin={'l': 0, 't': 0, 'b': 0, 'r': 0},
                mapbox={
                    'center': {'lon': 8.540192, 'lat': 47.378177},
                    'style': "open-street-map",
                    'zoom': 10})

            fig.show()

    def print_best_path(self):
        """Print the best path obtained."""

        paths_dataframe = self.paths_dataframe.copy()

        # path_dataframe is already sorted by level of confidence
        list_path_id = list(paths_dataframe.path_id.unique())
        if len(list_path_id) > 0:
            path = paths_dataframe[paths_dataframe['path_id']
                                   == list_path_id[0]]

            print("\n\nThe best route satisfying the level confidence {}% is the following:".format(
                self.confidence_widget.value))
            self.print_path(path)
            print("")

    def print_all_paths(self):
        """Print all the paths obtained, sorted by departure time and then confidence level."""

        paths_dataframe = self.paths_dataframe.copy()
        list_path_id = list(paths_dataframe.path_id.unique())
        if len(list_path_id) > 0:
            print(
                "\nHere is the list of all possible paths, sorted by departure time and then confidence level:")

        for path_id in list_path_id:
            path = paths_dataframe[paths_dataframe['path_id'] == path_id]
            print("\n#################################################\n")
            departure_time = path.iloc[0]['departure_time'][:5]
            path_probability = path.iloc[0]['path_probability'] * 100
            print("With departure time at {} and probability {}%:".format(
                departure_time, path_probability))
            self.print_path(path)

    def print_path(self, path):
        """Helper function to print a single path"""

        # Counter for number of changes
        i = 1
        for j, row in path.iterrows():
            # Current step is arrival i.e. target is reached
            if row['departure_time'] == "target":
                print("\n{}) Finally, you arrives at your destination ({}) at {}".format(
                    i, row['stop_name'], row['arrival_time'][:5]))

            # Current step is beginning of the journey.
            elif row['arrival_time'] == "source":

                # First ride is a walk
                if row['transport_type'] == "walking":
                    print("\n{}) You start your journey by walking to station {} at {}.".format(
                        i, path.loc[j + 1].stop_name, row['departure_time'][:5]))
                # First ride is a public transport
                else:
                    print("\n{}) You start your journey by taking the {} at {} at station {}.".format(
                        i, row['transport_type'], row['departure_time'][:5], row['stop_name']))

            # Current step is a change of transport type
            elif row['arrival_time']:
                # Quit transport to walk
                if row['transport_type'] == "walking":
                    print("\n{}) After your arrived at {} in station {}, you need to walk to station {}.".format(
                        i, row['arrival_time'][:5], row['stop_name'], path.loc[j + 1]['stop_name']))
                 # Change transport type for another type e.g. bus -> tram
                else:
                    print("\n{}) After your arrived at {} at station {}, you take the {} at {}.".format(
                        i, row['arrival_time'][:5], row['stop_name'], row['transport_type'], row['departure_time'][:5]))

            # Current step does not require any action i.e. stay in transport
            else:
                i -= 1
            i += 1

    def print_values_widget(self):
        """Print the values of the widget."""

        print(self.starting_station_widget.value)
        print(self.ending_station_widget.value)
        print(self.weekday_widget.value)
        print(self.arrival_time_widget.value)
        print(self.confidence_widget.value)
        print(self.num_path_on_map_widget.value)


def load_pickle(path_data):
    """
    Helper function to load pickle object.

    Parameters:
    ----------
    path_data : string
        Path to the file to be loaded
    Returns
    ----------
    var :
        The loaded object
    """
    with open(path_data, 'rb') as f:
        var = pickle.load(f)
    f.close()

    return var