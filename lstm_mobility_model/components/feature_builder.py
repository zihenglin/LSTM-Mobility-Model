import numpy as np
import tensorflow as tf

from lstm_mobility_model.config import Constants


class FeatureBuilder(object):
    """Feature builder
    """

    def __init__(self,
                 constants=None):
        """
        """
        self.constants = Constants() \
            if constants is None \
            else constants

    def build_time_feature_vector(self, time):
        """Return the time itself as continuous feature of the
        time.
        Args:
            time(tf.tensor): vectors of time the day.
        """
        return time

    def build_time_feature_vector_rule(self, time):
        """Build a time feature vector from time of day. The features include
        the following based on designed rules:
            start_weekend,
            start_morning_hour,
            start_lunch_hour,
            start_afternoon_hour,
            start_dinner_hour,
            start_home_hour
        Args:
            time(tf.tensor): vectors of time the day.
        Returns:
            (tf.tensor): feature vectors of time
        """
        start_morning_hour = self.is_morning_hour(time)
        start_lunch_hour = self.is_lunch_hour(time)
        start_afternoon_hour = self.is_afternoon_hour(time)
        start_dinner_hour = self.is_dinner_hour(time)
        start_home_hour = self.is_home_hour(time)

        if isinstance(start_home_hour, tf.Tensor):
            return tf.cast(
                tf.concat([
                    start_morning_hour, start_lunch_hour, start_afternoon_hour, start_dinner_hour,
                    start_home_hour
                ], axis=1), 'float32')

        else:
            return np.concatenate([
                start_morning_hour, start_lunch_hour, start_afternoon_hour, start_dinner_hour,
                start_home_hour
            ], 1).astype('float')

    def is_morning_hour(self, time_of_day):
        """Get binary indicators of whether a time
        of day is during "morning hour".
        Args:
            time_of_day(tf.tensor): time of day tensor.
        Return:
            (tf.tensor): the binary indicators tensor.
        """
        return (time_of_day >= self.constants.MORNING_HOUR_START /
                self.constants.DURATION_MAX) & \
            (time_of_day <= self.constants.MORNING_HOUR_END /
             self.constants.DURATION_MAX)

    def is_afternoon_hour(self, time_of_day):
        """Get binary indicators of whether a time
        of day is during "afternoon hour".
        Args:
            time_of_day(tf.tensor): time of day tensor.
        Return:
            (tf.tensor): the binary indicators tensor.
        """
        return (time_of_day >= self.constants.AFTERNOON_HOUR_START /
                self.constants.DURATION_MAX) & \
            (time_of_day <= self.constants.AFTERNOON_HOUR_END /
             self.constants.DURATION_MAX)

    def is_lunch_hour(self, time_of_day):
        """Get binary indicators of whether a time
        of day is during "lunch hour".
        Args:
            time_of_day(tf.tensor): time of day tensor.
        Return:
            (tf.tensor): the binary indicators tensor.
        """
        return (time_of_day >= self.constants.LUNCH_HOUR_START /
                self.constants.DURATION_MAX) & \
            (time_of_day <= self.constants.LUNCH_HOUR_END /
             self.constants.DURATION_MAX)

    def is_dinner_hour(self, time_of_day):
        """Get binary indicators of whether a time
        of day is during "dinner hour".
        Args:
            time_of_day(tf.tensor): time of day tensor.
        Return:
            (tf.tensor): the binary indicators tensor.
        """
        return (time_of_day >= self.constants.DINER_HOUR_START /
                self.constants.DURATION_MAX) & \
            (time_of_day <= self.constants.DINER_HOUR_END /
             self.constants.DURATION_MAX)

    def is_home_hour(self, time_of_day):
        """Get binary indicators of whether a time
        of day is during "home hour".
        Args:
            time_of_day(tf.tensor): time of day tensor.
        Return:
            (tf.tensor): the binary indicators tensor.
        """
        return time_of_day >= self.constants.HOME_HOUR_START / \
            self.constants.DURATION_MAX

    def calculate_distance(self, my_lat, my_lon, pts, is_mile=False):
        """Calculate the distance between two lat lng points.
        Should be moved elsewhere.
        """
        scale = 1.0
        if is_mile:
            scale = 0.000621371

        if isinstance(my_lat, tf.Tensor) or isinstance(pts, tf.Tensor):
            return tf.sqrt(((my_lat - pts[:, 0]) * 110000)**2 + ((my_lon - pts[:, 1]) * 90000) **
                           2) * scale
        else:
            return np.sqrt(((my_lat - pts[:, 0]) * 110000)**2 + ((my_lon - pts[:, 1]) * 90000) **
                           2) * scale
