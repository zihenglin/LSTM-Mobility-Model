class AbstractLstmModelBuilder(object):
    """The model combines all the components
    and provides interfaces for users to train the
    model and generate sequences with the model.
    """

    def _build_generating_model(self,
                                tensors):
        """Unroll and compile tensors in self.lstm_training
        together with self.tensors, self.mixture_density_builder
        and etc. The unrolled and compile tensors are used for
        training.
        """
        raise NotImplementedError

    def _build_training_model(self,
                              tensors):
        """Unroll and compile tensors in self.lstm_generating
        together with self.tensors, self.mixture_density_builder
        and etc. The unrolled and compile tensors are used for
        generating.
        """
        raise NotImplementedError

    def train(self,
              input_values_dict,
              epochs):
        """Train the lstm model with the unrolled and compiled
        tensors in self.lstm_training. The input and output
        sequences used in training are stored in input_values_dict.
        Args:
            input_values_dict(dict:(str -> np.array)): A dictionary
                of data with the keys being name as strings and
                values being the data as np.array.
            epochs(int): number of epochs that the model should
                be trained.
        """
        raise NotImplementedError

    def generate_complete_sequences(self,
                                    input_values_dict):
        """Generate the entire sequences with the input sequences
        given the input_values_dict.
        Args:
            input_values_dict(dict:(str -> np.array)): A dictionary
                of data with the keys being name as strings and
                values being the data as np.array.
        """
        raise NotImplementedError

    def generate_partial_sequences(self,
                                   input_values_dict):
        """Generate partially observed with the input sequences
        given the input_values_dict.
        Args:
            input_values_dict(dict:(str -> np.array)): A dictionary
                of data with the keys being name as strings and
                values being the data as np.array.
        """
        raise NotImplementedError

    def _save_model(self, tensorflow_session):
        """Save all weight on the graph that is realted to the
        tensorflow_session.
        tensorflow_session
        """
        raise NotImplementedError

    def _load_model(self, tensorflow_session):
        raise NotImplementedError
