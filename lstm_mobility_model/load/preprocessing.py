import os

import numpy as np
import pandas as pd

from lstm_mobility_model.config import (DataProcessingConstants,
                                        Features,
                                        OptionalFeatures,
                                        Constants)


class DataPreprocessor(object):
    """Data pre-processor is for filter, normalizing data
    based on rules. It also loads data from csv into
    formatted dataframes.
    """

    OUTPUT_COLUMN_NAMES = [Features.start_hour_since_day.name,
                           Features.duration.name,
                           Features.lat.name,
                           Features.lon.name,
                           Features.categorical_features.name,
                           OptionalFeatures.start_dayofweek.name,
                           OptionalFeatures.is_home.name,
                           OptionalFeatures.is_work.name,
                           OptionalFeatures.is_other.name]

    OUTPUT_COLUMN_NAMES_CATEGORICAL_LOCATION = \
        [Features.start_hour_since_day.name,
         Features.duration.name,
         OptionalFeatures.location_category.name,
         Features.categorical_features.name,
         OptionalFeatures.start_dayofweek.name,
         OptionalFeatures.is_home.name,
         OptionalFeatures.is_work.name,
         OptionalFeatures.is_other.name]

    def __init__(self,
                 required_columns=None):
        """
        Args:
            required_columns(list(str)): the required column names
                in the input dataframe.
        """
        self.required_columns = []

    def _check_column_names(self, traces_df):
        """Check if df has the required column name.
        Args:
            traces_df(DataFrame): a dataframe of stationary
                activities.
        """
        df_columns = set(traces_df.columns.values)
        for column_name in self.required_columns:
            assert column_name in df_columns

    def _add_time_features(self,
                           traces_df):
        """Get a unique day id from week_number and day of week
        and add to the input traces_df.
        Args:
            traces_df(DataFrame): a dataframe of stationary
                activities.
        Returns:
            (DataFrame): a dataframe of stationary
                activities.
        """
        start_index = pd.DatetimeIndex(traces_df['start_time'])
        end_index = pd.DatetimeIndex(traces_df['end_time'])

        if OptionalFeatures.start_date_id.name not in traces_df.columns.values:
            traces_df[OptionalFeatures.start_date_id.name] = start_index.date.astype(str)
            traces_df[OptionalFeatures.end_date_id.name] = end_index.date.astype(str)

        if Features.start_hour_since_day.name not in traces_df.columns.values:
            traces_df[Features.start_hour_since_day.name] = start_index.hour + \
                start_index.minute / 60.

        if OptionalFeatures.start_dayofweek.name not in traces_df.columns.values:
            traces_df[OptionalFeatures.start_dayofweek.name] = start_index.dayofweek

        return traces_df

    def filter_traces_dict(self, traces_dict):
        """Filter traces thatare too long or too short in traces_dict
        """
        return {key: dataframe
                for key, dataframe in traces_dict.items()
                if (len(dataframe) >= Constants.MINIMUM_DAILY_ACTIVITIES)
                and (len(dataframe) <= Constants.MAX_ACTIVITY_LEN)}

    def _normalize_data(self,
                        traces_df,
                        categorical_location=False):
        """Normalize spatial and temporal features so
        that the training of LSTM is more efficient.
        Args:
            traces_df(DataFrame): a dataframe of stationary
                activities.
        Returns:
            (DataFrame): a dataframe of stationary
                activities.
        """
        if traces_df is None or len(traces_df) < 1:
            return traces_df

        if not categorical_location:
            # Spatial features
            traces_df.loc[:, Features.lat.name] -= \
                DataProcessingConstants.LAT_CENTER
            traces_df.loc[:, Features.lon.name] -= \
                DataProcessingConstants.LON_CENTER
            traces_df.loc[:, Features.lat.name] /= \
                DataProcessingConstants.LAT_SCALE
            traces_df.loc[:, Features.lon.name] /= \
                DataProcessingConstants.LON_SCALE

        # Temporal features
        traces_df.loc[:, Features.start_hour_since_day.name] /= \
            DataProcessingConstants.START_TIME_MAX
        traces_df.loc[:, Features.duration.name] /= \
            DataProcessingConstants.DURATION_MAX

        # Dist to home and work
        if OptionalFeatures.dist_to_home.value in traces_df.columns.values and \
                OptionalFeatures.dist_to_work.value in traces_df.columns.values:
            traces_df.loc[:, OptionalFeatures.dist_to_home.value] /= \
                DataProcessingConstants.DIST_HOME_WORK_MAX
            traces_df.loc[:, OptionalFeatures.dist_to_work.value] /= \
                DataProcessingConstants.DIST_HOME_WORK_MAX
        return traces_df

    def _add_activity_category_labels(self, traces_df):
        """Add activity categorical labels based on dist to home and work.
        Args:
            traces_df(DataFrame): a dataframe of stationary
                activities.
        Returns:
            (DataFrame): a dataframe of stationary
                activities.
        """
        if traces_df is None or len(traces_df) < 1:
            return traces_df

        if OptionalFeatures.dist_to_home.value in traces_df.columns.values and \
                OptionalFeatures.dist_to_work.value in traces_df.columns.values:
            traces_df.loc[:, OptionalFeatures.is_home.value] = (
                traces_df[OptionalFeatures.dist_to_home.value] == 0).astype(int)
            traces_df.loc[:, OptionalFeatures.is_work.value] = (
                traces_df[OptionalFeatures.dist_to_work.value] == 0).astype(int)
            traces_df.loc[:, OptionalFeatures.is_other.value] = \
                (1 - traces_df[OptionalFeatures.is_home.value] -
                 traces_df[OptionalFeatures.is_work.value]).astype(int)

        assert OptionalFeatures.is_home.name in traces_df.columns.values
        assert OptionalFeatures.is_work.name in traces_df.columns.values
        assert OptionalFeatures.is_other.name in traces_df.columns.values
        traces_df.loc[:, Features.categorical_features.name] = \
            np.where(traces_df[[OptionalFeatures.is_home.value,
                                OptionalFeatures.is_work.value,
                                OptionalFeatures.is_other.value]].values)[1]
        return traces_df

    def _modify_start_time_duration(self, traces_df):
        """
        """
        traces_df = traces_df.reset_index()

        # Start time
        over_night_start_time = traces_df.iloc[0][Features.start_hour_since_day.name]
        traces_df.loc[0, Features.start_hour_since_day.name] = 0

        # Duration
        if traces_df.loc[0][OptionalFeatures.start_date_id.name] != \
                traces_df.loc[0][OptionalFeatures.end_date_id.name]:
            traces_df.loc[0, Features.duration.name] -= 24 - over_night_start_time
        else:
            traces_df.loc[0, Features.duration.name] += over_night_start_time

        return traces_df

    def preprocess_traces_to_dict(self, traces_df, **kwargs):
        """Process individual dataframe into dict of
        dataframe that has keys being date and values
        being the corresponding daily stationary activities.
        Args:
            traces_df(DataFrame): a dataframe of stationary
                activities.
        Returns:
            dict(int -> DataFrame): a dictionary of dataframe
                with key being the date id and values being
                the corresponding daily stationary activities.
        """

        traces_df = self._add_time_features(traces_df)

        unique_day_ids = set(traces_df[
            OptionalFeatures.start_date_id.name].values) | \
            set(traces_df[
                OptionalFeatures.end_date_id.name].values)

        traces_dict = {}
        for day_id in unique_day_ids:
            day_traces = traces_df[
                (traces_df[OptionalFeatures.start_date_id.name] == day_id) |
                (traces_df[OptionalFeatures.end_date_id.name] == day_id)]

            traces_dict[day_id] = self._modify_start_time_duration(day_traces)
        return traces_dict

    def preprocess_traces_df(self,
                             traces_df,
                             categorical_location=False,
                             **kwargs):
        """Perform preprocessing on an input traces_df.
        Args:
            traces_df(DataFrame): a dataframe of stationary
                activities.
        Returns:
            (DataFrame): a dataframe of stationary
                activities.
        """
        traces_df = self._add_time_features(traces_df)
        traces_df = self._normalize_data(traces_df, categorical_location)
        traces_df = self._add_activity_category_labels(traces_df)
        return self.filter_df_columns(traces_df,
                                      categorical_location,
                                      **kwargs)

    def filter_df_columns(self,
                          traces,
                          categorical_location=False,
                          keep_columns=[]):
        """Filter columns in traces dataframe that unused
        columns are deleted.
                Args:
            traces_df(DataFrame): a dataframe of stationary
                activities.
        Returns:
            (DataFrame): a dataframe of stationary
                activities.
        """
        if traces is None or len(traces) < 1:
            return traces

        if categorical_location:
            return traces[
                DataPreprocessor.OUTPUT_COLUMN_NAMES_CATEGORICAL_LOCATION +
                keep_columns]
        return traces[DataPreprocessor.OUTPUT_COLUMN_NAMES]

    def preprocess_traces_df_dict(self,
                                  traces_df_dict,
                                  categorical_location=False,
                                  **kwargs):
        """Preprocess dictionary of traces_df with key being date
        and values being corresponding dataframe.
        """
        output_dict = {}
        for key in traces_df_dict:
            preprocessed = self.preprocess_traces_df(
                traces_df_dict[key],
                categorical_location,
                **kwargs)
            if preprocessed is not None:
                output_dict[key] = preprocessed
        return output_dict

    def preprocess_traces_df_from_file_name(self,
                                            file_name,
                                            categorical_location=False):
        """Process data given a file_name.
        Args:
            file_name(str): the file name for the input
                csv file.
        Returns:
            (DataFrame): a dataframe of stationary
                activities.
        """
        dataframe = pd.read_csv(file_name)
        return self.preprocess_traces_df(dataframe,
                                         categorical_location)

    def preprocess_traces_df_from_file_folder(self,
                                              folder_name):
        """Process all csv files from a folder give the
        folder name.
        Args:
            file_name(str): the file name for the input
                csv file.
        Returns:
            (DataFrame): a dataframe of stationary
                activities.
        """
        file_list = os.listdir(folder_name)

        df_list = []
        for file_name in file_list:
            if file_name.endswith(".csv"):
                df_list.append(
                    self.preprocess_traces_df_from_file_name(
                        os.path.join(folder_name, file_name)
                    ))
        return pd.concat(df_list)
