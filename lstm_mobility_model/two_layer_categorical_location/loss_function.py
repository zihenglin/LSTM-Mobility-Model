import tensorflow as tf

from lstm_mobility_model.config import (Constants,
                                        Features,
                                        OptionalFeatures)
from lstm_mobility_model.components import (AbstractLossFunction,
                                            DistributionHelper)
from lstm_mobility_model.two_layer_latlng_location import TwoLayerLossFunction


class TwoLayerCategoricalLocationLossFunction(TwoLayerLossFunction):

    def get_loss(self,
                 n_location_categories,
                 layer_1_output_socres,
                 layer_2_output_socres):
        """Get total loss from layer 1 and layer 2 output."""
        batch_size = self.tensors.batch_size

        # Tensor shape: [batch_size, sequence_length]
        mask = self._get_mask()

        # Tensor shape: [batch_size, sequence_length]
        spatial_temporal_log_loss = self._get_spatial_temporal_loss(
            n_location_categories,
            layer_2_output_socres) * \
            mask
        # Tensor shape: [batch_size, sequence_length]
        categorical_log_loss = self._get_categorical_loss(layer_1_output_socres) * \
            mask

        # Tensor shape: []
        return [tf.reduce_sum(categorical_log_loss) / batch_size,
                tf.reduce_sum(spatial_temporal_log_loss) / batch_size]

    def _get_spatial_temporal_loss(self,
                                   n_location_categories,
                                   output_socres):
        """Compute the spatial-temporal log loss"""

        # Tensor shape: [batch_size, sequence_length, 1]
        target_start_time_sequence = self.tensors.get_placeholder_by_name(
            Features.start_hour_since_day.name)
        target_duration_sequence = self.tensors.get_placeholder_by_name(
            Features.duration.name)

        # Tensor shape: [batch_size, sequence_length, n_mixtures]
        target_location_category_sequence = tf.cast(
            self.tensors.get_placeholder_by_name(
                OptionalFeatures.location_category.name),
            tf.int32)
        target_location_category_one_hot = tf.reshape(tf.one_hot(
            target_location_category_sequence,
            axis=-1,
            depth=n_location_categories),
            [self.tensors.batch_size,
             Constants.INPUT_LENGTH,
             n_location_categories])

        # Tensor shape: [batch_size, sequence_length, n_mixtures]
        pi, mu_st, mu_dur, s_st, s_dur, rho_st_dur, pi_location = \
            self.mixture_density_builder.get_spatial_temporal_parameters(
                n_location_categories,
                output_socres)

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
        temporal_loss = tf.reduce_sum(pi * temporal_loss, axis=-1)

        # Tensor shape: [batch_size, sequence_length]
        location_loss = tf.reduce_sum(pi_location *
                                      target_location_category_one_hot,
                                      axis=-1)

        return -tf.log(tf.clip_by_value(temporal_loss * location_loss, 1e-30, 1e30))
