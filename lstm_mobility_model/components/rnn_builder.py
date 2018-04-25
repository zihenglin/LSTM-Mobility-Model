class AbstractRecurrentNeuralNetworkBuilder(object):
    """Build a recurrent neural network
    according to specified the input/output
    dimensions and number of unrolled steps.
    """

    DEFAULT_VARIABLE_SCOPE = "recurrent_neural_network"

    def __init__(self,
                 tensors,
                 mixture_density_output,
                 feature_builder=None,
                 variable_scope=None):
        """
        """
        self.tensors = tensors
        self.mixture_density_output = mixture_density_output
        self.feature_builder = feature_builder

        self.variable_scope = AbstractRecurrentNeuralNetworkBuilder.DEFAULT_VARIABLE_SCOPE \
            if variable_scope is None \
            else variable_scope

    def get_sequence_tensors(self):
        """Get sequence unrolled sequence unrolled tensors.
        """
        raise NotImplementedError

    def build_lstm(self):
        """Build lstm model by unrolling the lstm layer(s).
        Output the unrolled tensors that could be used for
        either generating sequences or training.
        """
        raise NotImplementedError

    def _get_current_samples(self,
                             current_output,
                             current_time):
        """Sample activity types, duration, travel time,
        next activity start time, and etc from current lstm
        output.
        Args:
            current_output(tf.tensor): the transformed output
                from rnn. Should have shape
                [batch_size, output_dimension]
            current_time(tf.tensor): the current time of each
                element in the batch. Should have shape
                [batch_size, 1]
        """
        raise NotImplementedError

    def _get_next_input(self,
                        current_time,
                        activity_type,
                        context_variables):
        """Get next lstm input features from sampled
        activity of previous step.
        """
        raise NotImplementedError

    def _save_model(self, file_path):
        """Save model weights to a file with location
        of file_path.
        Args:
            file_path(str): the file path string.
        """
        raise NotImplementedError

    def _load_model(self):
        """Load model weights to a file with location
        of file_path.
        Args:
            file_path(str): the file path string.
        """
        raise NotImplementedError
