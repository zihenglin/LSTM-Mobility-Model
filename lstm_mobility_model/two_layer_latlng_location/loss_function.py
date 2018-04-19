import tensorflow as tf

from lstm_mobility_model.config import (Constants,
                                        Features,
                                        OptionalFeatures)
from lstm_mobility_model.components import (AbstractLossFunction,
                                            DistributionHelper)


class TwoLayerLossFunction(AbstractLossFunction):

    def get_loss(self,
                 layer_1_output_socres,
                 layer_2_output_socres):
        """Get total loss from layer 1 and layer 2 output."""
        batch_size = self.tensors.batch_size

        # Tensor shape: [batch_size, sequence_length]
        mask = self._get_mask()

        # Tensor shape: [batch_size, sequence_length]
        spatial_temporal_log_loss = self._get_spatial_temporal_loss(layer_2_output_socres) * \
            mask
        # Tensor shape: [batch_size, sequence_length]
        categorical_log_loss = self._get_categorical_loss(layer_1_output_socres) * \
            mask

        # Tensor shape: []
        return [tf.reduce_sum(categorical_log_loss) / batch_size,
                tf.reduce_sum(spatial_temporal_log_loss) / batch_size]

    def _get_mask(self):
        """Get mask of shape [batch_size, sequence_length]"""

        return self.tensors.get_placeholder_by_name('mask')

    def _get_spatial_temporal_loss(self,
                                   output_socres):
        """Compute the log loss of each element [batch_size, sequence_length]"""

        # Tensor shape: [batch_size, sequence_length, 1]
        target_lat_sequence = self.tensors.get_placeholder_by_name(
            Features.lat.name)
        target_lon_sequence = self.tensors.get_placeholder_by_name(
            Features.lon.name)
        target_start_time_sequence = self.tensors.get_placeholder_by_name(
            Features.start_hour_since_day.name)
        target_duration_sequence = self.tensors.get_placeholder_by_name(
            Features.duration.name)

        # Tensor shape: [batch_size, sequence_length, n_mixtures]
        pi, mu_lat, mu_lon, s_lat, s_lon, \
            mu_st, mu_dur, s_st, s_dur, rho_st_dur = \
            self.mixture_density_builder.get_spatial_temporal_parameters(
                output_socres)

        # Tensor shape: [batch_size, sequence_length, n_mixtures]
        spatial_loss = \
            DistributionHelper.probability_2d_normal(target_lat_sequence,
                                                     target_lon_sequence,
                                                     mu_lat,
                                                     mu_lon,
                                                     s_lat,
                                                     s_lon)

        # Tensor shape: [batch_size, sequence_length, n_mixtures]
        temporal_loss = \
            DistributionHelper.probability_2d_normal_correlated(target_start_time_sequence,
                                                                target_duration_sequence,
                                                                mu_st,
                                                                mu_dur,
                                                                s_st,
                                                                s_dur,
                                                                rho_st_dur)

        # Tensor shape: [batch_size, sequence_length]
        loss = tf.reduce_sum(pi * spatial_loss * temporal_loss, axis=-1)

        return -tf.log(tf.clip_by_value(loss, 1e-30, 1e30))

    def _get_categorical_loss(self, categorical_sequences):
        """Compute the log loss of each element [batch_size, sequence_length]"""

        # Tensor shape: [batch_size, sequence_length, n_categories]
        target_indices = tf.cast(
            self.tensors.get_placeholder_by_name(
                Features.categorical_features.name),
            tf.int32)
        target_indices_one_hot = tf.reshape(tf.one_hot(
            target_indices,
            axis=-1,
            depth=Constants.NUMBER_OF_CATEGORIES),
            [self.tensors.batch_size,
             Constants.INPUT_LENGTH,
             Constants.NUMBER_OF_CATEGORIES])

        # Tensor shape: [batch_size, sequence_length, n_categories]
        categorical_distributions = \
            self.mixture_density_builder.get_categorical_parameters(
                categorical_sequences)

        # Tensor shape: [batch_size, sequence_length]
        categorical_loss = tf.reduce_sum(categorical_distributions *
                                         target_indices_one_hot,
                                         axis=-1)
        return -tf.log(categorical_loss)
