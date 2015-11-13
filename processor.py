import time
import numpy as np
import pandas as pd


ROW_LIMIT = 10000
COLUMNS = ["starttime", "stoptime", "start station id", "end station id"]
TARGET_STATION_ID = "529"


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

    def agg_by_period(self, df, col, new_col):
        """
        Convert the values in df.col to periods of length `self.period`
        and save in a new column called `new_col`.

        self.period = 5
        self.agg_by_period(df, 'start_time', 'start_period')
        -> new df with `start_period` column where 0 means first 5 minutes, 1 means between first 5 and first 10 minutes, etc
        """

        # Convert to timestamps
        df[col] = pd.to_datetime(df[col])

        # Convert int64index to datetime index
        # Group using timegrouper and create new column for start period and stop period
        df = df.set_index(pd.DatetimeIndex(df[col]))

        period_grouper = pd.TimeGrouper('{}Min'.format(self.period))

        # Group by `col` into periods
        group_by = df.groupby(period_grouper, as_index=False).apply(lambda x: x[col])

        df[new_col] = group_by.index.get_level_values(0)

        return df

    def agg_bike_flow_by_period(self, df, groupby_cols, output_col):
        """
        Creates new DataFrame stating how many bikes started/stopped in each period by station.
        """

        grouped_by = df.groupby(groupby_cols)
        bike_flow_df = pd.DataFrame({'count' : grouped_by.size()}).reset_index()

        # Rename columns
        bike_flow_df.columns = ["period", "station_id", output_col]
        return bike_flow_df

    def concat_bike_flow_dfs(self, start_df, stop_df):
        """
        Combine started and stopped bike flow dataframes.
        Ensure each station id has one row per period.
        """
        df_combined = pd.concat([start_df, stop_df]).fillna(0)

        # Let's group on period and station_id and then sum up along start_count and stop_count
        # And reset the multi-level index so we have a normal DataFrame
        merged_df = df_combined.groupby(['period', 'station_id']).sum().reset_index()
        return merged_df

    def pivot(self, df):
        """
        Creates a started/ended cols for each station id.
        Each row corresponds to snapshot of bike flow across all stations for a particular period.
        """
        # Let's construct a pivot table
        pivoted = pd.pivot_table(
            df,
            index=["period"],
            columns=["station_id"],
            aggfunc=np.sum,
            fill_value=0)

        # Go from multilevel index of start > station_id and stop > station_id to {station_id}_start and {station_id}_stop
        deeper_cols = pivoted.columns.get_level_values(1)
        top_level_cols = pivoted.columns.get_level_values(0)

        # Flattening the columns
        resultant_cols = []
        for i, station_id in enumerate(deeper_cols):
            if top_level_cols[i] == "start_count":
                resultant_cols.append("{}_{}".format(station_id, "out"))
            else:
                resultant_cols.append("{}_{}".format(station_id, "in"))
        pivoted.columns = resultant_cols
        pivoted = pivoted.reset_index()

        return pivoted

    def prepare_data(self, df):
        """
        Transforms the raw data into the format we want for classification:

        | Period | 173_in | 173_out | .... | 592_in | 592_out|
        |    0   |    1   |    2    | .....

        According to the above table, in period 0, 1 bike was docked at station 173 and 2 were taken out.
        """

        df = self.agg_by_period(df, 'starttime', 'start_period')
        df = self.agg_by_period(df, 'stoptime', 'stop_period')

        bike_start_df = self.agg_bike_flow_by_period(df, ['start_period', 'start station id'], 'start_count')
        bike_stop_df = self.agg_bike_flow_by_period(df, ['stop_period', 'end station id'], 'stop_count')

        df = self.concat_bike_flow_dfs(bike_start_df, bike_stop_df)

        return self.pivot(df)

    def separate_into_feature_and_target(self, df):
        """
        Separate data into feature data and target
        """
        X = df.copy()
        X.to_csv('data/processed_{}_rows_{}_bikes_out_{}_period.csv'.format(self.row_limit, self.bikes_out, self.period))
        # Turn number of out in station TARGET_STATION_ID into 1s and 0s
        answer_series = (
            df["{}_out".format(TARGET_STATION_ID) ] > self.bikes_out).apply(int)

        # Exclude the last period because we have no "right answer" to check against
        X = X[:-1]
        # Similarly, exclude first row of answer series to make sure we're aligned with our data set
        y = answer_series[1:]
        return {
            "data": X,
            "target": y
        }

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