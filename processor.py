import re
import time
import numpy as np
import pandas as pd
from utils import get_feature_and_target

ROW_LIMIT = 10000
COLUMNS = ["starttime", "stoptime", "start station id", "end station id"]
TIMESTAMP_FORMAT = '%m/%d/%Y %H:%M:%S'


class CBDataProcessor(object):
    """
    Transforms CitiBike systems data into the format we need for classification:

    | Period | 173_in | 173_out | .... | 592_in | 592_out|
    |    0   |    1   |    2    | .....

    According to the above table, in period 0, 1 bike was docked at station 173 and 2 were taken out.
    """
    def __init__(self, data_filename, bikes_out=1, row_limit=ROW_LIMIT, period=5, cols=COLUMNS):
        """
        We can modify the following:
        - raw data to use
        - number of bicycles taken out in the period to be considered in the positive class
        - number of rows we analyze
        - number of minutes in our period

        """
        self.data_filename = data_filename
        self.bikes_out = bikes_out
        self.row_limit = row_limit
        self.period = period
        self.cols = cols

    def read_data(self):
        """
        Reads in raw data from csv into DataFrame.
        """
        return pd.read_csv(self.data_filename, usecols=self.cols, nrows=self.row_limit)

    def convert_ts(self, df):
        """
        Convert dates to timestamp objects and sets that as index.
        """

        # Convert to timestamps (specify explicit format for speed)
        df['starttime'] = pd.to_datetime(df['starttime'], format=TIMESTAMP_FORMAT)
        df['stoptime'] = pd.to_datetime(df['stoptime'], format=TIMESTAMP_FORMAT)

        # Use start time as datetime index on dataframe. We need to do this to use TimeGrouper.
        df = df.set_index(pd.DatetimeIndex(df['starttime']))

        return df

    def agg_bike_flow_by_period(self, df, action):
        """
        Creates new DataFrame stating how many bikes started/stopped in each period by station.
        """

        # Make sure we don't modify the original since we're removing cols below
        df_raw = df.copy()

        # Group using timegrouper and create new column for start period and stop period
        period_grouper = pd.TimeGrouper('{}Min'.format(self.period))

        if action == 'started':
            station_col = 'start station id'
            del df_raw['end station id']
        else:
            station_col = 'end station id'
            del df_raw['start station id']

        # Group by period and station
        grouped = df_raw.groupby([period_grouper, station_col])

        # We no longer need these columns
        del df_raw['starttime']
        del df_raw['stoptime']

        # Gives # bikes started/stopped in each time period
        df_result = grouped.aggregate(np.size)

        # Each period becomes a row, and each station id a column
        unstacked = df_result.unstack(1).fillna(0)

        # Reset the index so that the period becomes a column (classifier expects this format)
        df_reset = unstacked.reset_index()

        # Rename the timestamp column period (defaults to index because was previous index)
        df_reset.rename(columns={'index': 'period'}, inplace=True)

        return df_reset

    def merge_flow_dfs(self, df_start, df_stop):
        """
        Joins the bike start/stop dataframes.
        """
        # Limit our analysis to stations that had at least one bike in each direction (in/out)
        # for the duration of our data set. The other stations must have had corrupt data or be
        # Citibike maintenance stations or something.
        in_stop_only = set(df_stop.columns).difference(df_start.columns)
        in_start_only = set(df_start.columns).difference(df_stop.columns)

        df_start.drop(in_start_only, axis=1, inplace=True)
        df_stop.drop(in_stop_only, axis=1, inplace=True)

        # We do outer join here so we include periods even if bikes only moved
        # in one direction for the duration of that period
        df_merged = pd.merge(
            df_stop,
            df_start,
            how="outer",
            on="period",
            suffixes=('_in', '_out'))

        # Exclude the first column: period for now
        flow_columns = df_merged.columns[1:]

        # Define regex for use in sort function to convert '72_in' -> 72
        non_digit = re.compile(r'[^\d]+')

        # Sort by the number of the station id, not alpabetically
        new_cols = sorted(flow_columns, key=lambda x: int(non_digit.sub('', x)))

        # We still have the period column display first
        new_cols.insert(0, 'period')

        df_merged = df_merged.reindex_axis(new_cols, axis=1)

        return df_merged

    def add_cyclicality(self, df):
        """
        Add the following columns to our data frame:
        - hour (0 - 23)
        - day of the week (0 - 6)
        - weekend? (True/False)
        - month (0 - 11)
        """
        df['hour'] = df['period'].apply(lambda x: x.hour)
        df['dayofweek'] = df['period'].apply(lambda x: x.dayofweek)
        df['is_weekend'] = df['period'].apply(lambda x: x.dayofweek in [5, 6])
        df['month'] = df['period'].apply(lambda x: x.month)

        return df

    def prepare_data(self, df):
        """
        Transforms the raw data into the format we want for classification:

        | Period | 173_in | 173_out | .... | 592_in | 592_out|
        |    0   |    1   |    2    | .....

        According to the above table, in period 0, 1 bike was docked at station 173 and 2 were taken out.
        """

        df = self.convert_ts(df)

        bike_start_df = self.agg_bike_flow_by_period(
            df, action='started')

        bike_stop_df = self.agg_bike_flow_by_period(
            df, action='stopped')

        df_merged = self.merge_flow_dfs(bike_start_df, bike_stop_df)

        df_cyclical = self.add_cyclicality(df_merged)

        return df_cyclical

    def separate_into_feature_and_target(self, df):
        """
        Separate data into feature data and target
        """
        df.to_csv('data/processed_{}_rows_{}_bikes_out_{}_period.csv'.format(self.row_limit, self.bikes_out, self.period))
        del df['period']
        return get_feature_and_target(df, self.bikes_out)

    def process(self):
        """
        1. Reads in data
        2. Pre-processes it for analysis
        3. Separates into feature data and target
        """
        start = time.time()
        df = self.read_data()
        df = self.prepare_data(df)

        result = self.separate_into_feature_and_target(df)
        end = time.time()
        result['duration'] = end - start
        return result
