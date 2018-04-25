import numpy as np
import tensorflow as tf

from lstm_mobility_model.config import Constants


class DistributionHelper(object):
    """Helper class for building distribution
    tensors from tensors of parameters.
    """
    @staticmethod
    def probability_normal(y, mu, sigma):
        """Get a 1D normal distribution tensor with
        mean and standard deviation tensors.
        Args:
            mu(tf.tensor): tensors of the mean.
            sigma(tf.tensor): tensors of the standard deviation.
        Return:
            (tf.tensor): normal distribution tensor
        """
        dist = tf.contrib.distributions.Normal(loc=mu,
                                               scale=sigma)
        return dist.prob(y)

    @staticmethod
    def probability_2d_normal(x1, x2, mu1, mu2, sigma1, sigma2):
        """Get the probability density of a 2D point given a 2D normal
        distribution without correlation and parameterized by
        mean, standard deviation.
        Args:
            x1(tf.tensor): data on the first dimension.
            x2(tf.tensor): data on the second dimension.
            mu1(tf.tensor): mean on the first dimension.
            mu2(tf.tensor): mean on the second dimension.
            sigma1(tf.tensor): standard deviation on the
                first dimension.
            sigma2(tf.tensor): standard deviation on the
                second dimension.
        Return:
            (tf.tensor): probability densities.
        """
        return DistributionHelper.probability_normal(x1, mu1, sigma1) * \
            DistributionHelper.probability_normal(x2, mu2, sigma2)

    @staticmethod
    def probability_2d_normal_correlated(x1,
                                         x2,
                                         mu1,
                                         mu2,
                                         sigma1,
                                         sigma2,
                                         rho):
        """Get the probability density of a 2D point given a 2D normal
        distribution parameterized by mean, standard
        deviation and correlation coefficient.
        Args:
            x1(tf.tensor): data on the first dimension.
            x2(tf.tensor): data on the second dimension.
            mu1(tf.tensor): mean on the first dimension.
            mu2(tf.tensor): mean on the second dimension.
            sigma1(tf.tensor): standard deviation on the
                first dimension.
            sigma2(tf.tensor): standard deviation on the
                second dimension.
            rho(tf.tensor): correlation coefficients.
        Return:
            (tf.tensor): probability densities.
        """
        mu = mu2 + rho * sigma2 / sigma1 * (x1 - mu1)
        s = tf.sqrt(1 - rho**2) * sigma2
        out = DistributionHelper.probability_normal(x1, mu1, sigma1) \
            * DistributionHelper.probability_normal(x2, mu, s)
        return out

    @staticmethod
    def get_softmax(unnormalized, dim=-1):
        """Normalize probabilities using softmax function.
        The normalization is along the last dimension.
        Args:
            unnormalized(tf.tensor): un-normalized tensors
                of probabilities.
        Returns:
            (tf.tensor): normalized tensors of probabilities.
        """
        normalized_prob = tf.clip_by_value(unnormalized,
                                           -20, 20)
        return tf.clip_by_value(tf.nn.softmax(normalized_prob, dim=dim),
                                1e-10,
                                1 - 1e-10)


class SamplingHelper(object):
    """A helper class for sampling from distributions.
    """

    @staticmethod
    def sample_multinomial_most_likely(probabilities, **kwargs):
        """Sample from multinomial distribution given probability
        tensors. probabilities should should have shape
        [batch_size, num_classes].
        Args:
            probabilities(tf.tensor): 2d probability tensors
                with shape [batch_size, num_classes].
        Returns:
            (tf.tensor): tensors of samples.
        """
        batch_size = probabilities.get_shape()[0].value
        return tf.reshape(tf.argmax(probabilities, axis=1),
                          [batch_size, 1])

    @staticmethod
    def sample_multinomial(probabilities, **kwargs):
        """Sample from multinomial distribution given probability
        tensors. probabilities should should have shape
        [batch_size, num_classes].
        Args:
            probabilities(tf.tensor): 2d probability tensors
                with shape [batch_size, num_classes].
        Returns:
            (tf.tensor): tensors of samples.
        """
        log_prob = tf.log(tf.clip_by_value(probabilities, 1e-30, 1 - 1e-30))
        return tf.multinomial(logits=log_prob, num_samples=1, **kwargs)

    @staticmethod
    def sample_multinomial_one_hot(probabilities, depth, **kwargs):
        """Sample from multinomial distribution given probability
        tensors. probabilities should should have shape
        [batch_size, num_classes].
        Args:
            probabilities(tf.tensor): 2d probability tensors
                with shape [batch_size, num_classes].
        Returns:
            (tf.tensor): tensors of samples.
        """
        indices = SamplingHelper.sample_multinomial(probabilities, **kwargs)
        indices = tf.reshape(indices, [-1])
        return tf.one_hot(indices=indices, depth=depth, axis=-1)

    @staticmethod
    def sample_multinomial_batch_index(probabilities,
                                       **kwargs):
        """Sample from multinomial distribution given probability
        tensors.
        Args:
            probabilities(tf.tensor): 2d probability tensors.
        Returns:
            (tf.tensor): index tensors as type tf.int32.
        """
        batch_size = probabilities.get_shape()[0].value
        batch_index = np.array(
            range(batch_size)).reshape(batch_size, 1)

        sampled_index = SamplingHelper.sample_multinomial(probabilities, **kwargs)

        return tf.cast(
            tf.concat([batch_index,
                       sampled_index], 1),
            tf.int32)

    @staticmethod
    def sample_from_1d_normal(mu, s, shape=[1], **kwargs):
        """Sample from 1D normal distribution
        given mean and standard deviations.
        Args:
            mu(tf.tensor): tensor of mean.
            s(tf.tensor): tensor of standard
                deviation.
        Returns:
            (tf.tensors): tensor of samples.
        """
        return tf.clip_by_value(
            tf.truncated_normal(shape, mu, s, **kwargs),
            -1e20, 1e20)

    @staticmethod
    def sample_from_1d_normal_mixture(pi, mu, s, **kwargs):
        """Sample from 1d normal distribution using
        mu, sigma and index.
        Args:
            pi(tf.tensor): tensor of mixture weights.
            mu(tf.tensor): tensor of mean.
            s(tf.tensor): tensor of standard
                deviation.
        Returns:
            (tf.tensors): tensor of samples.
        """
        batch_size = mu.get_shape()[0].value
        mixture_index = SamplingHelper.sample_multinomial_batch_index(pi,
                                                                      **kwargs)

        return tf.reshape(
            SamplingHelper.sample_from_1d_normal(tf.gather_nd(mu, mixture_index),
                                                 tf.gather_nd(s, mixture_index),
                                                 **kwargs),
            [batch_size, 1])

    @staticmethod
    def sample_mixture_2d_normal(pi,
                                 mu_1,
                                 mu_2,
                                 s_1,
                                 s_2,
                                 **kwargs):
        """Sample from 2d mixture normal distribution.
        Args:
            pi(tf.tensor): tensor of mixture weights.
            mu_1(tf.tensor): tensor of mean.
            mu_1(tf.tensor): tensor of mean.
            s_1(tf.tensor): tensor of standard
                deviation.
            s_2(tf.tensor): tensor of standard
                deviation.
        Returns:
            (tf.tensors): tensor of samples.
        """
        batch_size = pi.get_shape()[0].value
        mixture_index = SamplingHelper.sample_multinomial_batch_index(pi,
                                                                      **kwargs)

        return tf.reshape(SamplingHelper.sample_from_1d_normal(tf.gather_nd(mu_1, mixture_index),
                                                               tf.gather_nd(s_1, mixture_index),
                                                               **kwargs), [batch_size, 1]), \
            tf.reshape(SamplingHelper.sample_from_1d_normal(tf.gather_nd(mu_2, mixture_index),
                                                            tf.gather_nd(s_2, mixture_index),
                                                            **kwargs), [batch_size, 1])

    @staticmethod
    def sample_mixture_2d_normal_correlated(pi,
                                            mu_1,
                                            mu_2,
                                            s_1,
                                            s_2,
                                            rho,
                                            **kwargs):
        """Sample from 2d mixture normal distribution
        with correlations.
        Args:
            pi(tf.tensor): tensor of mixture weights.
            mu_1(tf.tensor): tensor of mean.
            mu_1(tf.tensor): tensor of mean.
            s_1(tf.tensor): tensor of standard
                deviation.
            s_2(tf.tensor): tensor of standard
                deviation.
            rho(tf.tensor): tensor of correlation
                coefficient.
        Returns:
            (tf.tensors): tensor of samples.
        """

        batch_size = pi.get_shape()[0].value
        mixture_index = SamplingHelper.sample_multinomial_batch_index(pi,
                                                                      **kwargs)

        mu_1 = tf.gather_nd(mu_1, mixture_index)
        mu_2 = tf.gather_nd(mu_2, mixture_index)
        s_1 = tf.gather_nd(s_1, mixture_index)
        s_2 = tf.gather_nd(s_2, mixture_index)
        rho = tf.gather_nd(rho, mixture_index)

        sample_1 = tf.clip_by_value(tf.truncated_normal([1], mu_1, s_1), -1e20, 1e20)

        mu_2_new = mu_2 + rho * s_2 / s_1 * (sample_1 - mu_1)
        s_2_new = tf.sqrt(1 - rho**2) * s_2

        sample_2 = tf.clip_by_value(tf.truncated_normal([1], mu_2_new, s_2_new), -1e20, 1e20)

        return tf.reshape(sample_1, [batch_size, 1]), \
            tf.reshape(sample_2, [batch_size, 1])

    @staticmethod
    def sample_mixture_2d_correlated_truncated(pi,
                                               mu_1,
                                               mu_2,
                                               s_1,
                                               s_2,
                                               rho,
                                               truncated_min,
                                               truncated_max,
                                               **kwargs):
        """Sample from 2d mixture normal distribution with correlations.
        Samples will be bounded by truncated_min and truncated_max.
        Now the truncation is implemented using tf.minimum and
        tf.maximum since tf.parameterized_truncated_normal is not
        documented and not working properly.
        Args:
            pi(tf.tensor): tensor of mixture weights.
            mu_1(tf.tensor): tensor of mean.
            mu_1(tf.tensor): tensor of mean.
            s_1(tf.tensor): tensor of standard
                deviation.
            s_2(tf.tensor): tensor of standard
                deviation.
            rho(tf.tensor): tensor of correlation
                coefficient.
            truncated_min(float): sampling truncation
                minmum value.
            truncated_max(float): sampling truncation
                maximum value.
        Returns:
            (tf.tensors): tensor of samples.
        """

        sample_1, sample_2 = \
            SamplingHelper.sample_mixture_2d_normal_correlated(pi,
                                                               mu_1,
                                                               mu_2,
                                                               s_1,
                                                               s_2,
                                                               rho,
                                                               **kwargs)
        sample_1 = tf.minimum(sample_1, truncated_max)
        sample_2 = tf.minimum(sample_2, truncated_max)
        sample_1 = tf.maximum(sample_1, truncated_min)
        sample_2 = tf.maximum(sample_2, truncated_min)
        return sample_1, sample_2

    @staticmethod
    def sample_mixture_2d_correlated_truncated_conditioned(x_1,
                                                           pi,
                                                           mu_1,
                                                           mu_2,
                                                           s_1,
                                                           s_2,
                                                           rho,
                                                           truncated_min,
                                                           truncated_max,
                                                           **kwargs):
        """Sample from 2nd component from 2d mixture normal
        distribution with the 1st component observed. Samples
        will be bounded by truncated_min and truncated_max.
        Now the truncation is implemented using tf.minimum and
        tf.maximum since tf.parameterized_truncated_normal is not
        documented and not working properly.
        Args:
            x_1(tf.tensor): observed value on the
                first dimension.
            pi(tf.tensor): tensor of mixture weights.
            mu_1(tf.tensor): tensor of mean.
            mu_1(tf.tensor): tensor of mean.
            s_1(tf.tensor): tensor of standard
                deviation.
            s_2(tf.tensor): tensor of standard
                deviation.
            rho(tf.tensor): tensor of correlation
                coefficient.
            truncated_min(float): sampling truncation
                minmum value.
            truncated_max(float): sampling truncation
                maximum value.
        Returns:
            (tf.tensors): tensor of samples.
        """
        batch_size = pi.get_shape()[0].value
        mixture_index = SamplingHelper.sample_multinomial_batch_index(pi,
                                                                      **kwargs)

        mu_1 = tf.gather_nd(mu_1, mixture_index)
        mu_2 = tf.gather_nd(mu_2, mixture_index)
        s_1 = tf.gather_nd(s_1, mixture_index)
        s_2 = tf.gather_nd(s_2, mixture_index)
        rho = tf.gather_nd(rho, mixture_index)

        mu_2_new = mu_2 + rho * s_2 / s_1 * (x_1 - mu_1)
        s_2_new = tf.sqrt(1 - rho**2) * s_2

        sample_2 = SamplingHelper.sample_truncated_1d_normal(
            mu_2_new, s_2_new, truncated_max, truncated_min, **kwargs)
        return tf.reshape(sample_2, [batch_size, 1])

    @staticmethod
    def sample_mixture_spatial_temporal_conditioned_time(
            observed_time,
            pi,
            mu_x,
            mu_y,
            s_x,
            s_y,
            mu_st,
            mu_dur,
            s_st,
            s_dur,
            rho_st_dur,
            duration_min=0.0035,
            duration_max=24,
            **kwargs):
        # TODO: Add tests for this function.
        """Sample from spatial temporal components from 4d
        mixture normal distribution with observed time. There
        are only correlations between the 2 temporal components
        (start time and duration). Samples
        will be bounded by truncated_min and truncated_max.
        Now the truncation is implemented using tf.minimum and
        tf.maximum since tf.parameterized_truncated_normal is not
        documented and not working properly.
        Args:

        Returns:
            (tf.tensors): tensor of samples.
        """
        batch_size = pi.get_shape()[0].value

        # Tensor shape: [batch_size, number_of_categories]
        updated_pi = pi * DistributionHelper.probability_normal(
            observed_time, mu_st, s_st)

        # updated_pi = pi

        # Tensor shape: [batch_size, 2]
        mixture_index = SamplingHelper.sample_multinomial_batch_index(
            updated_pi,
            **kwargs)

        # Tensor shape: [batch_size,]
        mu_x = tf.gather_nd(mu_x, mixture_index)
        mu_y = tf.gather_nd(mu_y, mixture_index)
        s_x = tf.gather_nd(s_x, mixture_index)
        s_y = tf.gather_nd(s_y, mixture_index)
        mu_st = tf.gather_nd(mu_st, mixture_index)
        mu_dur = tf.gather_nd(mu_dur, mixture_index)
        s_st = tf.gather_nd(s_st, mixture_index)
        s_dur = tf.gather_nd(s_dur, mixture_index)
        rho_st_dur = tf.gather_nd(rho_st_dur, mixture_index)

        # Tensor shape: [batch_size,]
        mu_dur_new = mu_dur + rho_st_dur * s_dur / s_st * \
            (tf.reshape(observed_time,  [-1]) - mu_st)
        s_dur_new = tf.sqrt(1 - rho_st_dur**2) * s_dur

        # Tensor shape: [batch_size,]
        sampled_duration = SamplingHelper.sample_truncated_1d_normal(
            mu_dur_new, s_dur_new, duration_max, duration_min, **kwargs)
        sampled_x = SamplingHelper.sample_from_1d_normal(
            mu_x, s_x)
        sampled_y = SamplingHelper.sample_from_1d_normal(
            mu_y, s_y)

        # Tensor shape: [batch_size, 1]
        return tf.reshape(sampled_x, [batch_size, 1]), \
            tf.reshape(sampled_y, [batch_size, 1]), \
            tf.reshape(sampled_duration, [batch_size, 1])

    @staticmethod
    def sample_truncated_1d_normal(mu,
                                   s,
                                   truncated_max,
                                   truncated_min,
                                   **kwargs):
        """Sample from parameterized truncated 1d normal.
        Now the truncation is implemented using tf.minimum and
        tf.maximum since tf.parameterized_truncated_normal is not
        documented and not working properly.
        Args:
            mu(tf.tensor): tensor of mean.
            s(tf.tensor): tensor of standard
                deviation.
            truncated_min(float): sampling truncation
                minmum value.
            truncated_max(float): sampling truncation
                maximum value.
        Returns:
            (tf.tensors): tensor of samples.
        """
        sampled_value = SamplingHelper.sample_from_1d_normal(mu, s,
                                                             **kwargs)
        sampled_value = tf.minimum(sampled_value, truncated_max)
        return tf.maximum(sampled_value, truncated_min)


class ParameterHelper(object):
    """Helper class for functions that transform
    neural network outputs into distribution
    parameters.
    """

    @staticmethod
    def get_pi(input_tensor,
               bias=None,
               dim=-1):
        """Transform un-normalized mixture component weights
        (probabilities) into normalized weights with bias. Higher
        bias will make high value weights even higher and low
        value weights even lower.
        Args:
            input_tensor(tf.tensor): un-normalized mixture component
                weights.
            bias(int): normalization bias.
            dim(int): normalized along which dimension of
                the input tensor.
        Returns:
            (tf.tensor): normalized mixture component weights.
        """
        if bias is not None:
            input_tensor *= 1 + bias
        return DistributionHelper.get_softmax(input_tensor, dim=dim)

    @staticmethod
    def get_non_negative(intput_tensor,
                         bias=None):
        """Transform tensors into non-negative values using
        exponential function. It is used in generating
        standard deviations of distributions. Use the bias
        to control the transformation.
        Args:
            input_tensor(tf.tensor): un-transformed tensors.
            bias(int): normalization bias.
        Returns:
            (tf.tensor): non-negative tensors.
        """
        # TODO: mo8ve values into constants
        if bias is not None:
            intput_tensor -= bias
        return tf.exp(tf.clip_by_value(intput_tensor,
                                       -10,
                                       10))

    @staticmethod
    def get_rho(intput_tensor):
        """Transform the input tensor into values between
        -1 and 1 exclusive. It is useful for transforming
        tensors into correlation coefficients.
        Args:
            input_tensor(tf.tensor): un-transformed tensors.
        Returns:
            (tf.tensor): tensors with values between -1 and 1.
        """
        # TODO: move values into constants
        return tf.clip_by_value(tf.tanh(intput_tensor),
                                -0.9999,
                                0.9999)

    @staticmethod
    def get_mu(intput_tensor):
        """Simply clip the tensor values in a range.
        Args:
            input_tensor(tf.tensor): input tensors.
        Returns:
            (tf.tensor): tensors with clipped values
                between a range.
        """
        # TODO: move values into constants
        return tf.clip_by_value(intput_tensor,
                                -1e5, 1e5)

    @staticmethod
    def get_finished(intput_tensor):
        """Transform tensors to be between 0 and 1 exclusive
        using sigmoid function.
        Args:
            input_tensor(tf.tensor): input tensors.
        Returns:
            (tf.tensor): tensors transformed by sigmoid
                function.
        """
        # TODO: move values into constants
        return tf.sigmoid(tf.clip_by_value(intput_tensor,
                                           -20, 20))
