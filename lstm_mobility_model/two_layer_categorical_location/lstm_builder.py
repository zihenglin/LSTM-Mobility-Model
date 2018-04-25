import tensorflow as tf


from lstm_mobility_model.config import (Constants,
                                        Features,
                                        OptionalFeatures)
from lstm_mobility_model.two_layer_latlng_location.lstm_builder import (LstmHelperFunctions,
                                                                        TwoLayerLSTMGenerating,
                                                                        TwoLayerLSTMTraining)


class TwoLayerCategoricalLocationLSTMTraining(
        TwoLayerLSTMTraining):
    """LSTM model for location choices as location ID.
        Model structure at training"""

    def build_lstm(self):
        """Build the lstm model for sequence generation"""
        with tf.variable_scope(self.variable_scope):

            # Initialize tensors
            lstm_layer_1 = self.tensors.get_lstm_layers_by_name('lstm_layer_1')
            lstm_layer_2 = self.tensors.get_lstm_layers_by_name('lstm_layer_2')
            current_time = self.tensors.get_placeholder_by_name(
                Features.start_hour_since_day.name)[:, 0, :]
            constex_features = self.tensors.get_placeholder_by_name(
                Features.contex_features.name)
            sampled_location = tf.cast(self.tensors.get_placeholder_by_name(
                OptionalFeatures.initial_location_category_input.name), tf.int64)

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
                lstm_input_layer_2 = LstmHelperFunctions.get_2nd_layer_categorical_location_input(
                    sampled_activity_types,
                    sampled_location,
                    current_time,
                    current_context_feature,
                    self.tensors.batch_size,
                    self.tensors.n_location_categories)

                # LSTM layer 2 input and output
                (lstm_output_layer_2, lstm_state_layer_2) = \
                    lstm_layer_2(lstm_input_layer_2,
                                 lstm_state_layer_2,
                                 scope='lstm_layer_2')

                # Load next activity start time
                next_activity_start_time, sampled_location = \
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

    def _sample_spatial_temporal(self,
                                 time_step,
                                 **kwargs):
        """Sample spatial and temporal. At training, we get the
            observation from next timestemp"""
        start_time_sequences = self.tensors.get_placeholder_by_name(
            Features.start_hour_since_day.name)
        if time_step == Constants.INPUT_LENGTH - 1:
            next_start_time = start_time_sequences[:, time_step, :]
        else:
            next_start_time = start_time_sequences[:, time_step + 1, :]

        location_sequences = self.tensors.get_placeholder_by_name(
            OptionalFeatures.location_category.name)
        location = tf.cast(location_sequences[:, time_step, :], tf.int64)

        # Add noise to activity start time
        next_start_time += tf.truncated_normal(
            shape=(self.tensors.batch_size, 1),
            stddev=0.01)

        return next_start_time, location

    def _sample_activity_location_categories(self,
                                             time_step,
                                             **kwargs):
        """Sample activity choices. At training, we get the
            observation from next timestemp"""
        activity_types = self.tensors.get_placeholder_by_name(
            Features.location_categories.name)
        return tf.cast(activity_types[:, time_step, :], tf.int32)


class TwoLayerCategoricalLocationLSTMGenerating(
        TwoLayerLSTMGenerating):
    """LSTM model for location choices as location ID.
        Model structure at testing"""

    DEFAULT_GENERATING_BATCH_SIZE = 1
    DEFAULT_SAMPLING_BIAS = 0.0

    def get_sequence_tensors(self):
        """Get all output tensors as a list"""
        return [self.generated_activity_start_time,
                self.generated_activity_duration,
                self.generated_activity_location,
                self.generated_activity_type]

    def build_lstm(self):
        """Build the lstm model"""
        with tf.variable_scope(self.variable_scope):

            # Initialize tensors
            lstm_layer_1 = self.tensors.get_lstm_layers_by_name('lstm_layer_1')
            lstm_layer_2 = self.tensors.get_lstm_layers_by_name('lstm_layer_2')
            current_time = tf.reshape(self.tensors.get_placeholder_by_name(
                Features.start_hour_since_day.name)[:, 0, :],
                [TwoLayerCategoricalLocationLSTMGenerating.DEFAULT_GENERATING_BATCH_SIZE, 1])
            constex_features = self.tensors.get_placeholder_by_name(
                Features.contex_features.name)
            sampled_location = tf.cast(self.tensors.get_placeholder_by_name(
                OptionalFeatures.initial_location_category_input.name), tf.int64)

            # Initial variables
            lstm_state_layer_1, lstm_state_layer_2 = LstmHelperFunctions.get_initial_state(
                lstm_layer_1,
                lstm_layer_2,
                TwoLayerCategoricalLocationLSTMGenerating.DEFAULT_GENERATING_BATCH_SIZE
            )
            corrected_activity_types = LstmHelperFunctions.get_initial_activity_types(
                self.tensors,
                TwoLayerCategoricalLocationLSTMGenerating.DEFAULT_GENERATING_BATCH_SIZE)

            generated_activity_start_time = []
            generated_activity_duration = []
            generated_activity_location = []
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
                    TwoLayerCategoricalLocationLSTMGenerating.DEFAULT_GENERATING_BATCH_SIZE)

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
                lstm_input_layer_2 = LstmHelperFunctions.get_2nd_layer_categorical_location_input(
                    corrected_activity_types,
                    sampled_location,
                    current_time,
                    current_context_feature,
                    TwoLayerCategoricalLocationLSTMGenerating.DEFAULT_GENERATING_BATCH_SIZE,
                    self.tensors.n_location_categories)

                # LSTM layer 2 input and output
                (lstm_output_layer_2, lstm_state_layer_2) = \
                    lstm_layer_2(lstm_input_layer_2,
                                 lstm_state_layer_2,
                                 scope='lstm_layer_2')

                # Sample spatial temporal
                sampled_location, sampled_duration,\
                    sampled_next_activity_start_time = \
                    self._sample_spatial_temporal(
                        lstm_output_layer_2,
                        current_time,
                        time_step,
                        bias=self.sampling_bias)

                # Store samples
                generated_activity_start_time.append(current_time)
                generated_activity_duration.append(sampled_duration)
                generated_activity_location.append(sampled_location)
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
        self.generated_activity_location = \
            tf.transpose(generated_activity_location, [1, 0, 2])
        self.generated_activity_type = \
            tf.transpose(generated_activity_type, [1, 0, 2])
        self.mixture_density_parameters_layer_1 = \
            tf.transpose(mixture_density_parameters_layer_1, [1, 0, 2])
        self.mixture_density_parameters_layer_2 = \
            tf.transpose(mixture_density_parameters_layer_2, [1, 0, 2])

    def _sample_spatial_temporal(self,
                                 lstm_output_layer_2,
                                 current_time,
                                 time_step,
                                 bias,
                                 **kwargs):
        """Sample location and duration from mixture density output."""
        # Sample spatial-temporal features
        lstm_output_layer_2 = tf.matmul(
            lstm_output_layer_2,
            self.tensors.get_trainable_variables_by_name('output_embedding_layer_2')) + \
            self.tensors.get_trainable_variables_by_name('output_bias_layer_2')
        sampled_location, sampled_duration = \
            self.mixture_density_output.sample_spatial_temporal(
                self.tensors.n_location_categories,
                lstm_output_layer_2,
                current_time,
                bias,
                **kwargs)

        travel_time = tf.random_uniform(shape=tf.shape(current_time),
                                        minval=0.005,
                                        maxval=0.015)
        sampled_next_activity_start_time = current_time + sampled_duration + travel_time
        sampled_location = tf.cast(sampled_location, tf.int64)
        return sampled_location, sampled_duration, sampled_next_activity_start_time

    def _update_sampled_activity_type(self,
                                      sampled_activity_type,
                                      time_step):
        """Correct the next time step feed to the LSTM model based on
        observation indices.
        """

        observed_activity_types = tf.cast(tf.reshape(self.tensors.get_placeholder_by_name(
            Features.categorical_features.name)[:, time_step, :],
            [TwoLayerCategoricalLocationLSTMGenerating.DEFAULT_GENERATING_BATCH_SIZE, 1]),
            tf.int64)
        observation_index = tf.reshape(self.tensors.get_placeholder_by_name(
            OptionalFeatures.is_observed.value)[:, time_step],
            [TwoLayerCategoricalLocationLSTMGenerating.DEFAULT_GENERATING_BATCH_SIZE, 1])
        observation_index = tf.cast(observation_index, tf.int64)

        corrected_activity_type = observation_index * observed_activity_types + \
            (1 - observation_index) * sampled_activity_type

        return corrected_activity_type

    def _correct_next_activity_start_time(self,
                                          sampled_next_activity_start_time,
                                          time_step):
        """Correct the next time step feed to the LSTM model based on
        observation indices.
        """
        time_step = min(Constants.INPUT_LENGTH - 1, time_step + 1)

        observed_next_activity_start_time = tf.reshape(self.tensors.get_placeholder_by_name(
            Features.start_hour_since_day.name)[:, time_step, :],
            [TwoLayerCategoricalLocationLSTMGenerating.DEFAULT_GENERATING_BATCH_SIZE, 1])

        observation_index = tf.reshape(self.tensors.get_placeholder_by_name(
            OptionalFeatures.is_observed.value)[:, time_step - 1],
            [TwoLayerCategoricalLocationLSTMGenerating.DEFAULT_GENERATING_BATCH_SIZE, 1])

        return observation_index * observed_next_activity_start_time + \
            (1 - observation_index) * sampled_next_activity_start_time
