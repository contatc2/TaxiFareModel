import pandas as pd
from TaxiFareModel.utils import simple_time_tracker

AWS_BUCKET_PATH = "s3://wagon-public-datasets/taxi-fare-train.csv"
LOCAL_PATH = "../../raw_data/train.csv"

DIST_ARGS = dict(start_lat="pickup_latitude",
                 start_lon="pickup_longitude",
                 end_lat="dropoff_latitude",
                 end_lon="dropoff_longitude")

@simple_time_tracker
def get_data(nrows=10000, local=False, **kwargs):
    '''returns a DataFrame with nrows from s3 bucket'''
    if local:
        path = LOCAL_PATH
    else:
        path = AWS_BUCKET_PATH
    df = pd.read_csv(path, nrows=nrows)
    return df


def clean_df(df, test=False):
    df = df.dropna(how='any', axis='rows')
    df = df[(df.dropoff_latitude != 0) | (df.dropoff_longitude != 0)]
    df = df[(df.pickup_latitude != 0) | (df.pickup_longitude != 0)]
    if "fare_amount" in list(df):
        df = df[df.fare_amount <= 4000]
        df = df[df.fare_amount > 0]
    df = df[df.passenger_count < 8]
    df = df[df.passenger_count > 0]
    df = df[df["pickup_latitude"].between(left=40, right=42)]
    df = df[df["pickup_longitude"].between(left=-74.3, right=-72.9)]
    df = df[df["dropoff_latitude"].between(left=40, right=42)]
    df = df[df["dropoff_longitude"].between(left=-74, right=-72.9)]
    return df


def df_optimized(df, verbose=True, **kwargs):
    """
    Reduces size of dataframe by downcasting numeircal columns
    :param df: input dataframe
    :param verbose: print size reduction if set to True
    :param kwargs:
    :return:
    """
    in_size = df.memory_usage(index=True).sum()
    for type in ["float", "integer"]:
        l_cols = list(df.select_dtypes(include=type))
        for col in l_cols:
            df[col] = pd.to_numeric(df[col], downcast=type)
            if type == "float":
                df[col] = pd.to_numeric(df[col], downcast="integer")
    out_size = df.memory_usage(index=True).sum()
    ratio = (1 - round(out_size / in_size, 2)) * 100
    GB = out_size / 1000000000
    if verbose:
        print("optimized size by {} % | {} GB".format(ratio, GB))
    return df


def infer_dtypes(path):
    """
    infer optimized dtypes for dataframe future dataframe csv loading
    :param path:
    :return: dict {"colname": dtype} to pass as argument to pd.read_csv
    """
    df = pd.read_csv(path, nrows=100)
    df_opt = df_optimized(df, verbose=False)
    dtypes = df_opt.dtypes
    colnames = dtypes.index
    types = [i.name for i in dtypes.values]
    column_types = dict(zip(colnames, types))
    return column_types


if __name__ == "__main__":
    params = dict(nrows=10000000,
                  upload=False,
                  # set to False to get data from GCP (Storage or BigQuery)
                  local=True,
                  optimize=True)
    df = get_data(**params)
    params["optimize"] = False
    df_2 = get_data(**params)
    m1 = df.memory_usage().sum()/1000000
    m2 = df_2.memory_usage().sum()/1000000
    print(m1, m2, m1 / m2)
    mm = pd.merge(df, df_2, on="key")
