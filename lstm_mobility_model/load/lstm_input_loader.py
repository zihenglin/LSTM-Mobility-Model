from collections import defaultdict

import numpy as np
import logging

from lstm_mobility_model.config import (DataProcessingConstants,
                                        Features,
                                        OptionalFeatures)
from lstm_mobility_model.load.preprocessing import DataPreprocessor


class LstmInputLoader(object):

    def __init__(self,
                 data_preprocessor=None,
                 file_loading_constants=None,
                 context_feature_columns=None,
                 load_weekend=False):
        """
        Args:
            data_preprocessor(DataPreprocessor): a
                DataPreprocessor object for data pre-
                processing.
            file_loading_constants(DataProcessingConstants):
                a DataProcessingConstants object that holds
                data processing constants.
            spatial_columns(list(str)): list of names of
                spatial feature columns.
            temporal_columns(list(str)): list of names of
                temporal feature columns.
            categorical_columns(list(str)): list of names of
                categorical feature columns.,
            context_feature_columns(list(str)): list of names of
                contextural feature columns.
            load_weekend(bool): whether to load weekend data or
                not.
        """
        self.categorical_columns = Features.categorical_features.value
        self.data_processor = DataPreprocessor() \
            if data_preprocessor is None \
            else data_preprocessor
        self.file_loading_constants = DataProcessingConstants() \
            if file_loading_constants is None \
            else file_loading_constants
        self.context_feature_columns = Features.contex_features.value \
            if context_feature_columns is None \
            else context_feature_columns

        # TODO: make use of load_weekend
        self.load_weekend = load_weekend

    def get_number_location_categories(self, traces_dict):
        """Get number of location choices for setting up
        tensors for models.
        Args:
            traces_dict (dict(int -> pd.DataFrame)): stationary
                activities by day.
        Returns:
            (int): number of unique places in traces_dict
        """
        return len(np.unique(traces_dict[OptionalFeatures.location_category.name].value))

    def get_lstm_features_from_traces_dict(self,
                                           traces_dict,
                                           features=None):
        """Get LSTM input sequences from a dictionary of
        daily stationary activities.
        Args:
            traces_dict (dict(int -> pd.DataFrame)): stationary
                activities by day.
        Returns:
            dict(int -> np.array(float)): binary indicators of whether
                the stationary activities are observed.
        """
        return self.get_lstm_features_from_partial_traces_dict(
            traces_dict, cut_time=None, features=features)

    def get_lstm_features_from_partial_traces_dict(self,
                                                   traces_dict,
                                                   cut_time=None,
                                                   features=None):
        """Get LSTM input sequences as partial observations
        from a dictionary of daily stationary activities.
        Args:
            traces_dict (dict(int -> pd.DataFrame)): stationary
                activities by day.
            cut_time_of_day(float): cutting time of the day between
                0 and 24.
        Returns:
            dict(int -> np.array(float)): binary indicators of whether
                the stationary activities are observed.
        """
        features = [f.name for f in Features] \
            if features is None \
            else features

        feature_dict = defaultdict(list)

        for date_key in traces_dict:

            day_traces = traces_dict[date_key]
            day_traces_len = len(day_traces)

            # Add LSTM input output features
            if day_traces_len < self.file_loading_constants.MINIMUM_DAILY_ACTIVITIES or \
                    day_traces_len > self.file_loading_constants.MAX_ACTIVITY_LEN:
                continue

            for feature in features:
                feature_dict[feature].append(
                    self._get_feature_function_by_name(
                        feature)(day_traces))

            # Add observation flags
            feature_dict[OptionalFeatures.is_observed.name].append(
                self._get_feature_function_by_name(
                    OptionalFeatures.is_observed.name)(day_traces,
                                                       cut_time))

        # Clean up the values to be np.array
        for date_key in feature_dict:
            feature_dict[date_key] = np.array(feature_dict[date_key])

        return dict(feature_dict)

    def _get_partial_observation_flags(self,
                                       day_traces,
                                       cut_time_of_day):
        """Get binary indicators of whether the stationary
        activities in day_traces are partially observed
        before the cut_time_of_day.
        """
        raise NotImplementedError

    def _get_observation_flags(self,
                               day_traces,
                               cut_time_of_day=None):
        """Get binary indicators of whether the stationary
        activities in day_traces are observed before the
        cut_time_of_day.
        Args:
            day_traces(pd.DataFrame): stationary activities of a
                day.
            cut_time_of_day(float): cutting time of the day between
                0 and 24.
        Returns:
            np.array(float): binary indicators of whether the
                stationary activities are observed.
        """
        start_hour_since_day = day_traces[Features.start_hour_since_day.name].values

        if cut_time_of_day is None:
            observation_flags = np.zeros_like(start_hour_since_day)
            return self._pad_sequence_first_dimension(
                observation_flags.astype(float)[:, np.newaxis])

        # Fix the start hour of first activity
        start_hour_since_day[0] = 0

        # duration = day_traces[Features.duration.name].values
        # observation_flags = (start_hour_since_day + duration) < \
        #     float(cut_time_of_day) / DataProcessingConstants.DURATION_MAX

        observation_flags = (start_hour_since_day) < \
            float(cut_time_of_day) / DataProcessingConstants.DURATION_MAX

        return self._pad_sequence_first_dimension(
            observation_flags.astype(float)[:, np.newaxis])

    def _get_latitude(self, day_traces):
        """Get latitude from day_traces.
        Args:
            day_traces(pd.DataFrame): stationary activities of a
                day.
        Returns:
            np.array(float): the padded latitude.
        """
        return self._pad_sequence_first_dimension(
            day_traces[Features.lat.name].values[:, np.newaxis]
        )

    def _get_longitude(self, day_traces):
        """Get longitude from day_traces.
        Args:
            day_traces(pd.DataFrame): stationary activities of a
                day.
        Returns:
            np.array(float): the padded longitude.
        """
        return self._pad_sequence_first_dimension(
            day_traces[Features.lon.name].values[:, np.newaxis]
        )

    def _get_start_time_of_activities(self, day_traces):
        """Get start time of activities from day_traces.
        Args:
            day_traces(pd.DataFrame): stationary activities of a
                day.
        Returns:
            np.array(float): the padded start time of day.
        """
        start_time_of_activities = self._pad_sequence_first_dimension(
            day_traces[Features.start_hour_since_day.name].values[:, np.newaxis]
        )

        # Fix the start time of first activities
        start_time_of_activities[0, :] = 0

        return start_time_of_activities

    def _get_duration(self, day_traces):
        """Get activity duration from day_traces.
        Args:
            day_traces(pd.DataFrame): stationary activities of a
                day.
        Returns:
            np.array(float): the padded duration.
        """
        duration = day_traces[Features.duration.name].values[:, np.newaxis]
        return self._pad_sequence_first_dimension(duration)

    def _get_start_time_of_day(self, day_traces):
        """Get start time of the first activity of the
        day.
        Args:
            day_traces(pd.DataFrame): stationary activities of a
                day.
        Returns:
            np.array(float): the start time of first activity
                formatted into a 2d array.
        """
        # TODO: not used anymore. Deprecate it.
        return np.array([day_traces.iloc[0][
            Features.start_hour_since_day.name]])

    def _get_masks(self, day_traces):
        """Get a list of indicator that indicates whether that a datapoint
        is observed or padded, ie. masks.
        Args:
            data_length(int): the length of data.
        Returns:
            (np.array): the masks.
        """
        data_length = len(day_traces)
        mask = np.zeros(self.file_loading_constants.MAX_ACTIVITY_LEN)
        mask[:data_length] = 1
        return mask

    def _get_initial_activity_type_input(self, day_traces):
        """Get initial dummy input to the LSTM model. In the current
        cast, the initial dummy input consists only the activity type
        index.
        Return:
            (np.array): The initial dummy activity type.
        """
        return np.zeros(1)

    def _get_contex_features(self,
                             day_traces):
        """Get contextural features from day_traces.
        Args:
            day_traces(pd.DataFrame): stationary activities of a
                day.
        Returns:
            np.array(float): the padded contextural features.
        """
        target_length = self.file_loading_constants.MAX_ACTIVITY_LEN
        context_features = \
            [[day_traces.iloc[-1][column]] * target_length
             for column in self.context_feature_columns]
        context_features = np.array(context_features).T

        return self._pad_sequence_first_dimension(context_features,
                                                  pad_mode='edge')

    def _get_spatial_features(self, day_traces):
        """Get spatial features from day_traces.
        Args:
            day_traces(pd.DataFrame): stationary activities of a
                day.
        Returns:
            np.array(float): the padded spatial features.
        """

        latitude = day_traces[Features.lat.name].values
        longitude = day_traces[Features.lon.name].values
        return latitude, longitude

    def _get_location_category(self, day_traces):
        """Get categorical location features from day_traces.
        Args:
            day_traces(pd.DataFrame): stationary activities of a
                day.
        Returns:
            np.array(float): the padded spatial features.
        """
        category_location = day_traces[[OptionalFeatures.location_category.name]].values
        category_location = self._pad_sequence_first_dimension(category_location)
        return category_location

    def _get_initial_location_category_input(self, day_traces):
        """Get categorical location features from day_traces.
        Args:
            day_traces(pd.DataFrame): stationary activities of a
                day.
        Returns:
            np.array(float): the padded spatial features.
        """
        return np.zeros(1)

    def _get_temporal_features(self, day_traces):
        """Get contextural temporal from day_traces.
        Args:
            day_traces(pd.DataFrame): stationary activities of a
                day.
        Returns:
            np.array(float): the padded temporal features.
        """
        start_time = day_traces[Features.start_hour_since_day.name].values
        duration = day_traces[Features.duration.name].values
        return start_time, duration

    def _get_categorical_features(self, day_traces):
        """Get categorical features from day_traces.
        Args:
            day_traces(pd.DataFrame): stationary activities of a
                day.
        Returns:
            np.array(float): the padded categorical features.
        """
        category_features = day_traces[self.categorical_columns].values
        category_features = self._pad_sequence_first_dimension(category_features)

        # Adding dummy activity types for Tensorflow learning purposes
        pad_ind = np.sum(category_features, axis=1) != 1
        category_features[pad_ind, -1] = 1

        return np.where(category_features)[1][:, np.newaxis]

    def _pad_sequence_first_dimension(self,
                                      input_array,
                                      target_length=None,
                                      pad_mode='constant'):
        """Padding a 2d numpy array along the first dimension.
        The allowed pad_mode is either 'constant' or 'edge'.
        Args:
            input_array(np.array(float)): 2d numpy array to
                be padded.
            target_length(int): the target length that the
                input_array should be padded along the 1st
                dimension.
            pad_mode(str): should be either 'constant'
                or 'edge'
        Returns:
            (np.array(float)): the padded 2d array.
        """
        target_length = self.file_loading_constants.MAX_ACTIVITY_LEN \
            if target_length is None \
            else target_length
        length_to_pad = target_length - len(input_array)
        assert length_to_pad >= 0
        if pad_mode == 'constant':
            return np.pad(input_array,
                          ((0, length_to_pad),
                           (0, 0)),
                          pad_mode,
                          constant_values=(0))
        elif pad_mode == 'edge':
            return np.pad(input_array,
                          ((0, length_to_pad),
                           (0, 0)),
                          pad_mode)
        else:
            logging.error("Unknown padding mode {}.".format(pad_mode))

    def _get_feature_function_by_name(self,
                                      feature_name):
        """Get feature extraction functions using
        feature names.
        Args:
            feature_name(str): feature names that are defined
                in Features.
        Returns:
            (function): the corresponding feature extraction
                function.
        """

        if feature_name == Features.categorical_features.name:
            return self._get_categorical_features
        elif feature_name == Features.contex_features.name:
            return self._get_contex_features
        elif feature_name == Features.lat.name:
            return self._get_latitude
        elif feature_name == Features.lon.name:
            return self._get_longitude
        elif feature_name == Features.start_hour_since_day.name:
            return self._get_start_time_of_activities
        elif feature_name == Features.duration.name:
            return self._get_duration
        elif feature_name == Features.initial_activity_type_input.name:
            return self._get_initial_activity_type_input
        elif feature_name == Features.mask.name:
            return self._get_masks
        elif feature_name == OptionalFeatures.is_observed.name:
            return self._get_observation_flags
        elif feature_name == OptionalFeatures.location_category.name:
            return self._get_location_category
        elif feature_name == OptionalFeatures.initial_location_category_input.name:
            return self._get_initial_location_category_input

        logging.warning(("_get_feature_function_by_name "
                         "got unknown feature_name '{}'").format(feature_name))
        return None
