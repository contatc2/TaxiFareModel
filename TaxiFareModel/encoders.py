import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from TaxiFareModel.utils import haversine_vectorized
from TaxiFareModel.data import DIST_ARGS, get_data, clean_data
import pygeohash as gh


class TimeFeaturesEncoder(BaseEstimator, TransformerMixin):
    """Extract the day of week (dow), the hour, the month and the year from a
    time column."""
    def __init__(self, time_column, time_zone_name='America/New_York'):
        self.time_column = time_column
        self.time_zone_name = time_zone_name

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        X_ = X.copy()
        X_.index = pd.to_datetime(X[self.time_column])
        X_.index = X_.index.tz_convert(self.time_zone_name)
        X_["dow"] = X_.index.weekday
        X_["hour"] = X_.index.hour
        X_["month"] = X_.index.month
        X_["year"] = X_.index.year
        return X_[['dow', 'hour', 'month', 'year']]


class DistanceTransformer(BaseEstimator, TransformerMixin):
    """Compute the haversine distance between two GPS points."""
    # def __init__(self)
        #          start_lat="pickup_latitude",
        #          start_lon="pickup_longitude",
        #          end_lat="dropoff_latitude",
        #          end_lon="dropoff_longitude"):
        # self.start_lat = start_lat
        # self.start_lon = start_lon
        # self.end_lat = end_lat
        # self.end_lon = end_lon

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        X_ = X.copy()
        X_["distance"] = haversine_vectorized(X_,**DIST_ARGS)
        return X_[['distance']]

class AddGeohash(BaseEstimator, TransformerMixin):

    def __init__(self, precision=6):
        self.precision = precision

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        X['geohash_pickup'] = X.apply(
            lambda x: gh.encode(x.pickup_latitude, x.pickup_longitude, precision=self.precision), axis=1)
        X['geohash_dropoff'] = X.apply(
            lambda x: gh.encode(x.dropoff_latitude, x.dropoff_longitude, precision=self.precision), axis=1)
        return X[['geohash_pickup', 'geohash_dropoff']]


if __name__ == "__main__":
    params = dict(nrows=1000,
                  upload=False,
                  local=False,  # set to False to get data from GCP (Storage or BigQuery)
                  optimize=False)
    # df = get_data(**params)
    # df = clean_df(df)
    # dir = Direction()
    # dist_to_center = DistanceToCenter()
    # X = dir.transform(df)
    # X2 = dist_to_center.transform(df)
