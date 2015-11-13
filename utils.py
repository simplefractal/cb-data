import pandas as pd
TARGET_STATION_ID = "529"


def get_feature_and_target(df, bikes_out):
    X = df.copy()
    # Turn number of out in station TARGET_STATION_ID into 1s and 0s
    answer_series = (
        df["{}_out".format(TARGET_STATION_ID) ] > bikes_out).apply(int)

    # Exclude the last period because we have no "right answer" to check against
    X = X[:-1]
    # Similarly, exclude first row of answer series to make sure we're aligned with our data set
    y = answer_series[1:]
    return {
        "data": X,
        "target": y,
        "size": y.size,
    }


def get_data_from_file(row_limit, bikes_out, period):
    """
    Reads in data from data/processed_{row_limit}_bikes_{bikes_out}_period_{period}.csv
    and separate into feature data and target
    """
    file_name = 'data/processed_{}_rows_{}_bikes_out_{}_period.csv'.format(
        row_limit, bikes_out, period)
    return get_feature_and_target(
        pd.read_csv(file_name),
        bikes_out)
