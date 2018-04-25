import tensorflow as tf

from lstm_mobility_model.components import (AbstractMixtureDensityBuilder,
                                            DistributionHelper,
                                            SamplingHelper,
                                            ParameterHelper)
from lstm_mobility_model.two_layer_latlng_location import (TwoLayerMixtureDensityBuilder)


class TwoLayerCategoricalLocationMixtureDensityBuilder(
        TwoLayerMixtureDensityBuilder):
    """Mixture density builders for 2 layer structure. 1st layer
    outputs categorical parameters. Second layer outputs
    spatial temporal mixture distribution parameters.
    """

    NUMBER_OF_MIXTURE_PARAMETERS = 6
    MAX_STANDARD_DEVIATION = 10

    def sample_categorical(self,
                           neural_network_output_categorical,
                           **kwargs):
        """Sample activity type from categorical distribution."""
        categorical_distribution = DistributionHelper.get_softmax(
            neural_network_output_categorical)
        return SamplingHelper.sample_multinomial(categorical_distribution,
                                                 **kwargs)

    def sample_spatial_temporal(self,
                                n_location_categories,
                                neural_network_output_spatial,
                                current_time,
                                bias=None,
                                **kwargs):
        """Sample location and duration from mixture density output."""
        pi, mu_st, mu_dur, s_st, s_dur, rho_st_dur, pi_location = \
            self.get_spatial_temporal_parameters(
                n_location_categories,
                neural_network_output_spatial,
                bias=bias)

        sampled_duration = \
            SamplingHelper.sample_mixture_2d_correlated_truncated_conditioned(
                current_time, pi, mu_st, mu_dur, s_st, s_dur, rho_st_dur, 0, 1,
                **kwargs)

        sampled_location = SamplingHelper.sample_multinomial(pi_location,
                                                             **kwargs)

        return sampled_location, sampled_duration

    def get_categorical_parameters(self, neural_network_output):
        """Get categorical distribution for model loss."""
        return DistributionHelper.get_softmax(neural_network_output)

    def get_spatial_temporal_parameters(self,
                                        n_location_categories,
                                        neural_network_output,
                                        bias=None):
        """Get spatial adn temporal parameters for model loss."""
        if len(neural_network_output.shape) == 2:
            pi_location = neural_network_output[:, :n_location_categories]
            temporal_tensors = neural_network_output[:, n_location_categories:]
        else:
            pi_location = neural_network_output[:, :, :n_location_categories]
            temporal_tensors = neural_network_output[:, :, n_location_categories:]

        pi, mu_st, mu_dur, s_st, s_dur, rho_st_dur = \
            tf.split(
                temporal_tensors,
                TwoLayerCategoricalLocationMixtureDensityBuilder.NUMBER_OF_MIXTURE_PARAMETERS,
                axis=-1)

        pi = ParameterHelper.get_pi(pi, bias=bias)
        mu_dur = ParameterHelper.get_mu(mu_dur)
        mu_st = ParameterHelper.get_mu(mu_st)
        s_st = tf.clip_by_value(
            ParameterHelper.get_non_negative(s_st, bias=bias),
            0, TwoLayerCategoricalLocationMixtureDensityBuilder.MAX_STANDARD_DEVIATION)
        s_dur = tf.clip_by_value(
            ParameterHelper.get_non_negative(s_dur, bias=bias),
            0, TwoLayerCategoricalLocationMixtureDensityBuilder.MAX_STANDARD_DEVIATION)

        pi_location = ParameterHelper.get_pi(pi_location, bias=bias)
        rho_st_dur = ParameterHelper.get_rho(rho_st_dur)
        return pi, mu_st, mu_dur, s_st, s_dur, rho_st_dur, pi_location
