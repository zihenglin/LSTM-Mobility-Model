import tensorflow as tf

from lstm_mobility_model.components import AbstractRecurrentNeuralNetworkBuilder
from lstm_mobility_model.config import (Constants,
                                        Features,
                                        OptionalFeatures)


class LstmHelperFunctions(object):

    @staticmethod
    def get_initial_activity_types(tensors, batch_size):
        """Get initial activity type"""
        return tf.reshape(tf.cast(tensors.get_placeholder_by_name(
            Features.initial_activity_type_input.name), tf.int32),
            [batch_size, 1])

    @staticmethod
    def get_1st_or_2nd_layer_input(sampled_activity_types_indices,
                                   current_time,
                                   constex_features,
                                   batch_size):
        """Combine activity type, current time and contex features
        to form input to lstm layers.
        """
        activity_type_features = \
            tf.reshape(
                tf.one_hot(sampled_activity_types_indices,
                           axis=-1,
                           depth=Constants.NUMBER_OF_CATEGORIES),
                shape=[batch_size,
                       Constants.NUMBER_OF_CATEGORIES])

        return tf.concat([activity_type_features,
                          current_time,
                          constex_features],
                         axis=-1)

    @staticmethod
    def get_2nd_layer_categorical_location_input(
            sampled_activity_types_indices,
            sampled_location_categories,
            current_time,
            constex_features,
            batch_size,
            n_location_categories):
        """Combine activity type, current time and contex features
        to form input to lstm layers. For categorical location only.
        """
        activity_type_features = \
            tf.reshape(
                tf.one_hot(sampled_activity_types_indices,
                           axis=-1,
                           depth=Constants.NUMBER_OF_CATEGORIES),
                shape=[batch_size,
                       Constants.NUMBER_OF_CATEGORIES])

        location_categories = \
            tf.reshape(
                tf.one_hot(sampled_location_categories,
                           axis=-1,
                           depth=n_location_categories),
                shape=[batch_size,
                       n_location_categories])

        return tf.concat([activity_type_features,
                          location_categories,
                          current_time],
                         axis=-1)

    @staticmethod
    def get_initial_state(lstm_layer_1,
                          lstm_layer_2,
                          batch_size):
        """Get lstm initial states"""
        return lstm_layer_1.zero_state(batch_size,
                                       tf.float32),  \
            lstm_layer_2.zero_state(batch_size,
                                    tf.float32)


class TwoLayerLSTMTraining(
        AbstractRecurrentNeuralNetworkBuilder):

    def __init__(self,
                 tensors,
                 mixture_density_output,
                 feature_builder,
                 variable_scope=None):
        AbstractRecurrentNeuralNetworkBuilder.__init__(
            self,
            tensors,
            mixture_density_output,
            feature_builder,
            variable_scope
        )

        self.mixture_density_parameters_layer_1 = None
        self.mixture_density_parameters_layer_2 = None

    def get_sequence_tensors(self):
        """Get lstm output for computing loss"""
        return [self.mixture_density_parameters_layer_1,
                self.mixture_density_parameters_layer_2]

    def build_lstm(self):
        """Build the lstm model for training"""
        with tf.variable_scope(self.variable_scope):

            # Initialize tensors
            lstm_layer_1 = self.tensors.get_lstm_layers_by_name('lstm_layer_1')
            lstm_layer_2 = self.tensors.get_lstm_layers_by_name('lstm_layer_2')
            current_time = self.tensors.get_placeholder_by_name(
                Features.start_hour_since_day.name)[:, 0, :]
            constex_features = self.tensors.get_placeholder_by_name(
                Features.contex_features.name)

            # Initial variables
            lstm_state_layer_1, lstm_state_layer_2 = LstmHelperFunctions.get_initial_state(
                lstm_layer_1,
                lstm_layer_2,
                self.tensors.batch_size)
            sampled_activity_types = LstmHelperFunctions.get_initial_activity_types(
                self.tensors,
                self.tensors.batch_size)

            mixture_density_parameters_layer_1 = []
            mixture_density_parameters_layer_2 = []

            for time_step in range(Constants.INPUT_LENGTH):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()

                # Get current context feature
                current_context_feature = constex_features[:, time_step, :]

                # Current time step layer 1 input
                lstm_input_layer_1 = LstmHelperFunctions.get_1st_or_2nd_layer_input(
                    sampled_activity_types,
                    current_time,
                    current_context_feature,
                    self.tensors.batch_size)

                # LSTM layer 1 input and output
                (lstm_output_layer_1, lstm_state_layer_1) = \
                    lstm_layer_1(lstm_input_layer_1,
                                 lstm_state_layer_1,
                                 scope='lstm_layer_1')

                # Sample next activity
                sampled_activity_types = \
                    self._sample_activities_types(time_step)

                # Get 2nd layer input
                lstm_input_layer_2 = LstmHelperFunctions.get_1st_or_2nd_layer_input(
                    sampled_activity_types,
                    current_time,
                    current_context_feature,
                    self.tensors.batch_size)

                # LSTM layer 2 input and output
                (lstm_output_layer_2, lstm_state_layer_2) = \
                    lstm_layer_2(lstm_input_layer_2,
                                 lstm_state_layer_2,
                                 scope='lstm_layer_2')

                # Load next activity start time
                next_activity_start_time = \
                    self._sample_spatial_temporal(time_step)

                # Transform parameters
                lstm_output_layer_1 = tf.matmul(
                    lstm_output_layer_1,
                    self.tensors.get_trainable_variables_by_name('output_embedding_layer_1')) + \
                    self.tensors.get_trainable_variables_by_name('output_bias_layer_1')

                lstm_output_layer_2 = tf.matmul(
                    lstm_output_layer_2,
                    self.tensors.get_trainable_variables_by_name('output_embedding_layer_2')) + \
                    self.tensors.get_trainable_variables_by_name('output_bias_layer_2')

                # Store parameters
                mixture_density_parameters_layer_1.append(lstm_output_layer_1)
                mixture_density_parameters_layer_2.append(lstm_output_layer_2)

                # Update time
                current_time = next_activity_start_time

        self.mixture_density_parameters_layer_1 = \
            tf.transpose(mixture_density_parameters_layer_1, [1, 0, 2])
        self.mixture_density_parameters_layer_2 = \
            tf.transpose(mixture_density_parameters_layer_2, [1, 0, 2])

    def _sample_activities_types(self,
                                 time_step,
                                 **kwargs):
        """At training, get next timestamp activity type"""
        activity_types = self.tensors.get_placeholder_by_name(
            Features.categorical_features.name)
        return tf.cast(activity_types[:, time_step, :], tf.int32)

    def _sample_spatial_temporal(self,
                                 time_step,
                                 **kwargs):
        """At training, get next timestamp activity start time"""
        start_time_sequences = self.tensors.get_placeholder_by_name(
            Features.start_hour_since_day.name)

        if time_step == Constants.INPUT_LENGTH - 1:
            next_start_time = start_time_sequences[:, time_step, :]
        else:
            next_start_time = start_time_sequences[:, time_step + 1, :]

        next_start_time += tf.truncated_normal(
            shape=(self.tensors.batch_size, 1),
            stddev=0.01)
        return next_start_time


class TwoLayerLSTMGenerating(
        AbstractRecurrentNeuralNetworkBuilder):

    DEFAULT_GENERATING_BATCH_SIZE = 1
    DEFAULT_SAMPLING_BIAS = 0.0

    def __init__(self,
                 tensors,
                 mixture_density_output,
                 feature_builder,
                 sampling_bias=None,
                 variable_scope=None):
        AbstractRecurrentNeuralNetworkBuilder.__init__(
            self,
            tensors,
            mixture_density_output,
            feature_builder,
            variable_scope
        )

        # Generated Sequence
        self.generated_activity_start_time = None
        self.generated_activity_duration = None
        self.generated_activity_lat = None
        self.generated_activity_lon = None
        self.generated_activity_type = None

        # Sequence
        self.mixture_density_parameters_layer_1 = None
        self.mixture_density_parameters_layer_2 = None

        self.sampling_bias = TwoLayerLSTMGenerating.DEFAULT_SAMPLING_BIAS \
            if sampling_bias is None \
            else sampling_bias

    def get_sequence_tensors(self):
        """Get tensors of lstm output for sequence generation"""
        return [self.generated_activity_start_time,
                self.generated_activity_duration,
                self.generated_activity_lat,
                self.generated_activity_lon,
                self.generated_activity_type]

    def build_lstm(self):
        """Build the lstm model for sequence generation"""
        with tf.variable_scope(self.variable_scope):

            # Initialize tensors
            lstm_layer_1 = self.tensors.get_lstm_layers_by_name('lstm_layer_1')
            lstm_layer_2 = self.tensors.get_lstm_layers_by_name('lstm_layer_2')
            current_time = tf.reshape(self.tensors.get_placeholder_by_name(
                Features.start_hour_since_day.name)[:, 0, :],
                [TwoLayerLSTMGenerating.DEFAULT_GENERATING_BATCH_SIZE, 1])
            constex_features = self.tensors.get_placeholder_by_name(
                Features.contex_features.name)

            # Initial variables
            lstm_state_layer_1, lstm_state_layer_2 = LstmHelperFunctions.get_initial_state(
                lstm_layer_1,
                lstm_layer_2,
                TwoLayerLSTMGenerating.DEFAULT_GENERATING_BATCH_SIZE
            )
            corrected_activity_types = LstmHelperFunctions.get_initial_activity_types(
                self.tensors,
                TwoLayerLSTMGenerating.DEFAULT_GENERATING_BATCH_SIZE)

            generated_activity_start_time = []
            generated_activity_duration = []
            generated_activity_lat = []
            generated_activity_lon = []
            generated_activity_type = []

            mixture_density_parameters_layer_1 = []
            mixture_density_parameters_layer_2 = []

            for time_step in range(Constants.INPUT_LENGTH):
                # if time_step > 0:
                tf.get_variable_scope().reuse_variables()

                # Get current context feature
                current_context_feature = \
                    constex_features[:, time_step, :]

                # Current time step layer 1 input
                lstm_input_layer_1 = LstmHelperFunctions.get_1st_or_2nd_layer_input(
                    corrected_activity_types,
                    current_time,
                    current_context_feature,
                    TwoLayerLSTMGenerating.DEFAULT_GENERATING_BATCH_SIZE)

                # LSTM layer 1 input and output
                (lstm_output_layer_1, lstm_state_layer_1) = \
                    lstm_layer_1(lstm_input_layer_1,
                                 lstm_state_layer_1,
                                 scope='lstm_layer_1')

                # Sample next activity
                sampled_activity_types = \
                    self._sample_activities_types(lstm_output_layer_1,
                                                  time_step)

                # Correct sampled activity based on
                corrected_activity_types = self._update_sampled_activity_type(
                    sampled_activity_types,
                    time_step)

                # Get 2nd layer input
                lstm_input_layer_2 = LstmHelperFunctions.get_1st_or_2nd_layer_input(
                    corrected_activity_types,
                    current_time,
                    current_context_feature,
                    TwoLayerLSTMGenerating.DEFAULT_GENERATING_BATCH_SIZE)

                # LSTM layer 2 input and output
                (lstm_output_layer_2, lstm_state_layer_2) = \
                    lstm_layer_2(lstm_input_layer_2,
                                 lstm_state_layer_2,
                                 scope='lstm_layer_2')

                # Sample spatial temporal
                sampled_lat, sampled_lon, sampled_duration,\
                    sampled_next_activity_start_time = \
                    self._sample_spatial_temporal(
                        lstm_output_layer_2,
                        current_time,
                        time_step,
                        bias=self.sampling_bias)

                # Store samples
                generated_activity_start_time.append(current_time)
                generated_activity_duration.append(sampled_duration)
                generated_activity_lat.append(sampled_lat)
                generated_activity_lon.append(sampled_lon)
                generated_activity_type.append(sampled_activity_types)

                # Transform parameters
                lstm_output_layer_1 = tf.matmul(
                    lstm_output_layer_1,
                    self.tensors.get_trainable_variables_by_name('output_embedding_layer_1')) + \
                    self.tensors.get_trainable_variables_by_name('output_bias_layer_1')

                lstm_output_layer_2 = tf.matmul(
                    lstm_output_layer_2,
                    self.tensors.get_trainable_variables_by_name('output_embedding_layer_2')) + \
                    self.tensors.get_trainable_variables_by_name('output_bias_layer_2')

                # Store parameters
                mixture_density_parameters_layer_1.append(lstm_output_layer_1)
                mixture_density_parameters_layer_2.append(lstm_output_layer_2)

                # Update current time according to observations
                current_time = self._correct_next_activity_start_time(
                    sampled_next_activity_start_time,
                    time_step)

        self.generated_activity_start_time = \
            tf.transpose(generated_activity_start_time, [1, 0, 2])
        self.generated_activity_duration = \
            tf.transpose(generated_activity_duration, [1, 0, 2])
        self.generated_activity_lat = \
            tf.transpose(generated_activity_lat, [1, 0, 2])
        self.generated_activity_lon = \
            tf.transpose(generated_activity_lon, [1, 0, 2])
        self.generated_activity_type = \
            tf.transpose(generated_activity_type, [1, 0, 2])
        self.mixture_density_parameters_layer_1 = \
            tf.transpose(mixture_density_parameters_layer_1, [1, 0, 2])
        self.mixture_density_parameters_layer_2 = \
            tf.transpose(mixture_density_parameters_layer_2, [1, 0, 2])

    def _sample_activities_types(self,
                                 lstm_output_layer_1,
                                 time_step,
                                 **kwargs):
        """Sample activity type from 1s layer output"""
        lstm_output_layer_1 = tf.matmul(
            lstm_output_layer_1,
            self.tensors.get_trainable_variables_by_name('output_embedding_layer_1')) + \
            self.tensors.get_trainable_variables_by_name('output_bias_layer_1')
        sampled_activity_types = \
            self.mixture_density_output.sample_categorical(
                lstm_output_layer_1)

        return sampled_activity_types

    def _sample_spatial_temporal(self,
                                 lstm_output_layer_2,
                                 current_time,
                                 time_step,
                                 **kwargs):
        """Sample location and duration from 2nd layer output"""
        lstm_output_layer_2 = tf.matmul(
            lstm_output_layer_2,
            self.tensors.get_trainable_variables_by_name('output_embedding_layer_2')) + \
            self.tensors.get_trainable_variables_by_name('output_bias_layer_2')
        sampled_lat, sampled_lon, sampled_duration = \
            self.mixture_density_output.sample_spatial_temporal(
                lstm_output_layer_2,
                current_time,
                **kwargs)

        travel_time = tf.random_uniform(shape=tf.shape(current_time),
                                        minval=0.005,
                                        maxval=0.02)
        sampled_next_activity_start_time = current_time + sampled_duration + travel_time
        return sampled_lat, sampled_lon, sampled_duration, sampled_next_activity_start_time

    def _update_sampled_activity_type(self,
                                      sampled_activity_type,
                                      time_step):
        """For partially sequence prediction problem, correct the next timestamp
        activity type fed to the LSTM model based on observation indices.
        """
        observed_activity_types = tf.cast(tf.reshape(self.tensors.get_placeholder_by_name(
            Features.categorical_features.name)[:, time_step, :],
            [TwoLayerLSTMGenerating.DEFAULT_GENERATING_BATCH_SIZE, 1]),
            tf.int64)
        observation_index = tf.reshape(self.tensors.get_placeholder_by_name(
            OptionalFeatures.is_observed.value)[:, time_step],
            [TwoLayerLSTMGenerating.DEFAULT_GENERATING_BATCH_SIZE, 1])
        observation_index = tf.cast(observation_index, tf.int64)

        corrected_activity_type = observation_index * observed_activity_types + \
            (1 - observation_index) * sampled_activity_type

        return corrected_activity_type

    def _correct_next_activity_start_time(self,
                                          sampled_next_activity_start_time,
                                          time_step):
        """For partially sequence prediction problem, correct the next timestamp
        activity start time fed to the LSTM model based on observation indices.
        """
        time_step = min(Constants.INPUT_LENGTH - 1, time_step + 1)

        observed_next_activity_start_time = tf.reshape(self.tensors.get_placeholder_by_name(
            Features.start_hour_since_day.name)[:, time_step, :],
            [TwoLayerLSTMGenerating.DEFAULT_GENERATING_BATCH_SIZE, 1])

        observation_index = tf.reshape(self.tensors.get_placeholder_by_name(
            OptionalFeatures.is_observed.value)[:, time_step - 1],
            [TwoLayerLSTMGenerating.DEFAULT_GENERATING_BATCH_SIZE, 1])

        return observation_index * observed_next_activity_start_time + \
            (1 - observation_index) * sampled_next_activity_start_time
