import tensorflow as tf

from lstm_mobility_model.components import (AbstractMixtureDensityBuilder,
                                            DistributionHelper,
                                            SamplingHelper,
                                            ParameterHelper)


class TwoLayerMixtureDensityBuilder(
        AbstractMixtureDensityBuilder):
    """Mixture density builders for 2 layer structure. 1st layer
    outputs categorical parameters. Second layer outputs
    spatial temporal mixture distribution parameters.
    """

    NUMBER_OF_MIXTURE_PARAMETERS = 10

    def sample_categorical(self,
                           neural_network_output_categorical,
                           **kwargs):
        """Sample from categorial distribution"""
        categorical_distribution = DistributionHelper.get_softmax(
            neural_network_output_categorical)
        return SamplingHelper.sample_multinomial(categorical_distribution,
                                                 **kwargs)

    def sample_spatial_temporal(self,
                                neural_network_output_spatial,
                                current_time,
                                bias=None,
                                **kwargs):
        """Sample location and  categorial distribution"""
        pi, mu_lat, mu_lon, s_lat, s_lon, \
            mu_st, mu_dur, s_st, s_dur, rho_st_dur = \
            self.get_spatial_temporal_parameters(
                neural_network_output_spatial,
                bias=bias)

        sampled_lat, sampled_lon, sampled_duration = \
            SamplingHelper.sample_mixture_spatial_temporal_conditioned_time(
                current_time, pi, mu_lat, mu_lon, s_lat,
                s_lon, mu_st, mu_dur, s_st, s_dur, rho_st_dur,
                **kwargs)

        return sampled_lat, sampled_lon, sampled_duration

    def get_categorical_parameters(self, neural_network_output):
        """Get probabilities for computing loss"""
        return DistributionHelper.get_softmax(neural_network_output)

    def get_spatial_temporal_parameters(self, neural_network_output, bias=None):
        """Get mixture distributions for computing loss"""
        MAX_STANDARD_DEVIATION = 10

        pi, mu_lat, mu_lon, s_lat, s_lon, \
            mu_st, mu_dur, s_st, s_dur, rho_st_dur = \
            tf.split(
                neural_network_output,
                TwoLayerMixtureDensityBuilder.NUMBER_OF_MIXTURE_PARAMETERS,
                axis=-1)

        pi = ParameterHelper.get_pi(pi, bias=bias)
        mu_lat = ParameterHelper.get_mu(mu_lat)
        mu_lon = ParameterHelper.get_mu(mu_lon)
        mu_dur = ParameterHelper.get_mu(mu_dur)
        mu_st = ParameterHelper.get_mu(mu_st)

        s_lat = tf.clip_by_value(ParameterHelper.get_non_negative(s_lat, bias=bias),
                                 0, MAX_STANDARD_DEVIATION)
        s_lon = tf.clip_by_value(ParameterHelper.get_non_negative(s_lon, bias=bias),
                                 0, MAX_STANDARD_DEVIATION)
        s_st = tf.clip_by_value(ParameterHelper.get_non_negative(s_st, bias=bias),
                                0, MAX_STANDARD_DEVIATION)
        s_dur = tf.clip_by_value(ParameterHelper.get_non_negative(s_dur, bias=bias),
                                 0, MAX_STANDARD_DEVIATION)

        rho_st_dur = ParameterHelper.get_rho(rho_st_dur)

        return pi, mu_lat, mu_lon, s_lat, s_lon, \
            mu_st, mu_dur, s_st, s_dur, rho_st_dur
