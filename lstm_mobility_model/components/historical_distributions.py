import glob
import os

import tensorflow as tf
import numpy as np


class HistoricalTemporalDistribution(object):
    """Historical temporal distributions which are loaded from npy files.
    """

    DEFAULT_DURATION_MATRIX_FILES = ['Home_weekday', 'Work', 'Other_weekday']
    DEFAULT_TEMORAL_SCALE = 4
    MAXIMUM_START_HOUR = 23.9  # Hour

    def __init__(self,
                 duration_matrices):
        """
        Args:
            duration_matrices(tf.tensor): start time - duration 2d histograms
                with first index being activity type and 2nd index being
                starting hour.
        """
        self.duration_matrices = duration_matrices

    @staticmethod
    def from_duration_matrix_files(duration_matrix_path,
                                   duration_matrix_files=None):
        """
        Build a HistoricalTemporalDistribution Load joint dist of st and duration
        matrices into a tf.tensor.
        Args:
            duration_matrix_path(str): file path for duration matraces.
            duration_matrix_files(list(str)): list of names of npy files.
        """
        duration_matrix_files = \
            HistoricalTemporalDistribution.DEFAULT_DURATION_MATRIX_FILES \
            if duration_matrix_files is None \
            else duration_matrix_files

        duration_matrices = []
        for file_name in duration_matrix_files:
            duration_matrices.append(np.load(os.path.join(duration_matrix_path,
                                                          file_name + '.npy')))

        return HistoricalTemporalDistribution(
            tf.Variable(np.array(duration_matrices),
                        dtype='float',
                        name='duration_matrix'))

    def _get_flat_transformed_start_hour(self,
                                         start_hour,
                                         max_start_hour=MAXIMUM_START_HOUR):
        """Get flat start hour from start_hour as N-D tensor. The hours are
        multiplied by HistoricalTemporalDistribution.DEFAULT_TEMORAL_SCALE
        and casted to int to match the indices in self.duration_matrices.
        Args:
            start_hour(tf.tensor): N-D tensor of activity start time.
            max_start_hour(number): the maximum possible start hour
                of the day.
        Returns:
            (tf.tensor): the flattened and transformed start hour.
        """
        start_hour = tf.reshape(tf.minimum(start_hour, max_start_hour), [-1])
        start_hour *= HistoricalTemporalDistribution.DEFAULT_TEMORAL_SCALE
        return tf.cast(start_hour, 'int32')

    def _get_gathered_duration_matrices(self,
                                        flat_activity_type_indices):
        """Gather self.duration_matrices using
        flat_activity_type_indices.
        Args:
            flat_activity_type_indices(tf.tensor): should be 1-D
                tensor of activity indices.
        Returns:
            (tf.tensor): the flattened and transformed start hour.
        """
        return tf.gather(self.duration_matrices,
                         flat_activity_type_indices)

    def _get_duration_distributions(self,
                                    flat_transformed_start_hour,
                                    gathered_duration_matrices):
        """Get descrete distribution of durations from gathered_duration_matrices
        as a 3d matrix and flat_start_hour as indices.
        Args:
            flat_transformed_start_hour(tf.tensor):flattened and transformed
                start hour of activities which can serve as indices of
                flat_duration_matrices.
            gathered_duration_matrices(tf.tensor): gathered duration
                matrices, which has the same 1st dimension as
                flat_transformed_start_hour
        Returns:
            (tf.tensor):
        """
        total_data_length = flat_transformed_start_hour.get_shape()[0].value
        range_indices = np.array(range(total_data_length))[:, np.newaxis]
        flat_transformed_start_hour = tf.reshape(
            flat_transformed_start_hour, [total_data_length, 1])
        start_hour_2d_indices = tf.concat([range_indices,
                                           flat_transformed_start_hour],
                                          axis=1)
        return tf.gather_nd(gathered_duration_matrices, start_hour_2d_indices)

    def sample_duration(self,
                        activity_type_ind,
                        start_hour,
                        minimum_duration=0,
                        **kwargs):
        """Sample activity duration from historical temporal distributions
        given start hour and activity type. If the sampled duration is lower
        than the minimum_duration, the values will be replaced by
        minimum_duration.
        Args:
            activity_type_ind(tf.tensor): activity type indicators as
                0, 1, 2, ... n. Not one-hot.
            start_hour(tf.tensor): start hour of activities.
            minimum_duration(float): minimum duration of activities.
        Returns:
            (tf.tensor): sampled activity duration.
        """
        flat_activity_type_ind = tf.cast(tf.reshape(activity_type_ind, [-1]), 'int32')
        gathered_duration_matrices = self._get_gathered_duration_matrices(flat_activity_type_ind)

        flat_start_hour = self._get_flat_transformed_start_hour(start_hour)

        duration_distributions = self._get_duration_distributions(flat_start_hour,
                                                                  gathered_duration_matrices)

        sampled_duration_transformed = tf.multinomial(
            tf.log(duration_distributions), 1, **kwargs)

        sampled_duration = tf.cast(sampled_duration_transformed, 'float32') / \
            HistoricalTemporalDistribution.DEFAULT_TEMORAL_SCALE

        sampled_duration = tf.maximum(sampled_duration, minimum_duration)
        return tf.reshape(sampled_duration, start_hour.get_shape())


class HistoricalSpatialDistribution(object):
    """Historical temporal distributions which are loaded from npy files.
    """
    # TODO: Finish implementation if necessary.

    def __init__(self):
        """
        """
        pass

    @staticmethod
    def load_d_file(d_path):
        """
        Joint dist of dh and dw
        """
        D_total = {}
        os.chdir(d_path)
        for fi in glob.glob("*.npy"):
            typ = fi.split('.npy')[0]
            D_total[typ] = np.load(fi)

        return {'Other': D_total['Other_weekday']}

    def sample_distance_to_home_work(self,
                                     activity_type):
        """
        """
        pass
