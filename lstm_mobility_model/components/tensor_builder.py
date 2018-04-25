import tensorflow as tf
from tensorflow.python.ops import rnn_cell
import numpy as np

from lstm_mobility_model.config import Constants


class TensorBuilder(object):
    """Build helper tensors such as place holders,
    embeddings according to the input and output
    dimensions.
    """
    DEFAULT_STANDARD_DEVIATION = 0.025
    DEFAULT_LSTM_DROPOUT = 0.05
    DEFAULT_BATCH_SIZE = 100

    def __init__(self,
                 init_standard_deviation=None,
                 lstm_dropout=None,
                 batch_size=None):
        """
        Args:
            placeholders_dimension_enum,
            trainable_dimension_enum,
            nontrainable_dimension_enum,
            init_standard_deviation
        """

        # Tensorflow tunings
        self._init_standard_deviation = \
            TensorBuilder.DEFAULT_STANDARD_DEVIATION \
            if init_standard_deviation is None \
            else init_standard_deviation

        self.lstm_dropout = TensorBuilder.DEFAULT_LSTM_DROPOUT \
            if lstm_dropout is None \
            else lstm_dropout

        self.batch_size = TensorBuilder.DEFAULT_BATCH_SIZE \
            if batch_size is None \
            else batch_size

        # tensors fields
        self.placeholders = None
        self.trainable_variables = None
        self.nontrainable_variables = None
        self.lstm_layers = None

    def get_placeholder_by_name(self, name):
        """Get placeholders by their names.
        Return None if there is no such tenor by
        the name.
        Args:
            name(str): name of the tensor
        Returns:
            (tf.tensor): the tensor with the
                name.
        """
        return None if self.placeholders is None \
            else self.placeholders.get(name, None)

    def get_trainable_variables_by_name(self, name):
        """Get trainable variable by their names.
        Return None if there is no such tenor by
        the name.
        Args:
            name(str): name of the tensor
        Returns:
            (tf.tensor): the tensor with the
                name.
        """

        return None if self.trainable_variables is None \
            else self.trainable_variables.get(name, None)

    def get_nontrainable_variables_by_name(self, name):
        """Get non-trainable by their names.
        Return None if there is no such tenor by
        the name.
        Args:
            name(str): name of the tensor
        Returns:
            (tf.tensor): the tensor with the
                name.
        """
        return None if self.nontrainable_variables is None \
            else self.nontrainable_variables.get(name, None)

    def get_lstm_layers_by_name(self, name):
        """Get lstm layers by their names.
        Return None if there is no such tenor by
        the name.
        Args:
            name(str): name of the tensor
        Returns:
            (tf.tensor): the tensor with the
                name.
        """
        return None if self.lstm_layers is None \
            else self.lstm_layers.get(name, None)

    def build_lstm_layers(self,
                          dimension_dict):
        """Build LSTM layers with a dimension dictionary
        that has keys of strings and values of list of
        numbers representing dimensions of tensors
        Args:
            dimension_dict(dict:(str -> list(int))):
                a map of tensor name to dimensions.
        Return:
            dict:(str -> tf.tensor): a map of tensor
                name to tensorflow placeholder objects
        """
        output_dict = {}
        for name in dimension_dict:
            output_dict[name] = \
                self._build_tensorflow_lstm_units(
                    dimension=dimension_dict[name],
                    name=name)

        self.lstm_layers = output_dict

    def build_placeholders(self,
                           dimension_dict):
        """Build placeholders with a dimension dictionary
        that has keys of strings and values of list of
        numbers representing dimensions of tensors
        Args:
            dimension_dict(dict:(str -> list(int))):
                a map of tensor name to dimensions.
        Return:
            dict:(str -> tf.tensor): a map of tensor
                name to tensorflow placeholder objects
        """
        output_dict = {}
        for name in dimension_dict:
            output_dict[name] = \
                self._build_tensorflow_placeholder(
                    dimension=dimension_dict[name],
                    name=name)
        self.placeholders = output_dict

    def build_trainable_variables(self,
                                  dimension_dict):
        """Build trainable tensorflow variables using
        a map from tensor name to dimensions.
        Args:
            dimension_dict(dict:(str -> list(int))):
                a map of tensor name to dimensions.
        Return:
            dict:(str -> tf.tensor): a map of tensor
                name to tensorflow trainable variable
                objects
        """
        output_dict = {}
        for name in dimension_dict:
            output_dict[name] = \
                self._build_tensorflow_variable_trainable(
                    dimension=dimension_dict[name],
                    name=name)
        self.trainable_variables = output_dict

    def build_nontrainable_variables(self, dimension_dict):
        """Build trainable tensorflow variables using
        a map from tensor name to dimensions.
        Args:
            dimension_dict(dict:(str -> list(int))):
                a map of tensor name to dimensions.
        Return:
            dict:(str -> tf.tensor): a map of tensor
                name to tensorflow trainable variable
                objects
        """
        output_dict = {}
        for name in dimension_dict:
            output_dict[name] = \
                self._build_tensorflow_variable_nontrainable(
                    dimension=dimension_dict[name],
                    name=name)
        self.nontrainable_variables = output_dict

    def _build_tensorflow_placeholder(self,
                                      dimension=None,
                                      name=None,
                                      data_type='float',):
        """Build a single tensorflow placeholder using
        dimension, name and data type.
        Args:
            dimension(list(int)): tensor dimensions.
            name(str): tensor name.
            data_type(str): tensorflow data type.
        Return:
            (tf.tensor): tensorflow placeholder
        """
        return tf.placeholder(data_type,
                              dimension,
                              name)

    def _build_tensorflow_variable_trainable(self,
                                             dimension=None,
                                             name=None,
                                             data_type='float'):
        """Build a single tensorflow trainable variable using
        dimension, name and data type.
        Args:
            dimension(list(int)): tensor dimensions.
            name(str): tensor name.
            data_type(str): tensorflow data type.
        Return:
            (tf.tensor): tensorflow trainable variable
        """
        return tf.Variable(
            tf.truncated_normal(dimension,
                                stddev=self._init_standard_deviation,
                                dtype=data_type),
            name=name)

    def _build_tensorflow_variable_nontrainable(self,
                                                dimension=None,
                                                name=None):
        """Build a single tensorflow nontrainable variables using
        dimension, name and data type. The variables are
        initialized to all zero.
        Args:
            dimension(list(int)): tensor dimensions.
            name(str): tensor name.
            data_type(str): tensorflow data type.
        Return:
            (tf.tensor): tensorflow nontrainable variable
        """
        return tf.Variable(np.zeros(dimension),
                           dtype='float',
                           trainable=False,
                           name=name)

    def _build_tensorflow_lstm_units(self,
                                     dimension,
                                     name=None):
        """Build multi-cell LSTM units with dropout
        wrapper.
        Args:
            name(str): tensor name
        Returns:
            (tf.tensor) the tensor
        """

        return tf.nn.rnn_cell.MultiRNNCell(
            [rnn_cell.DropoutWrapper(
                rnn_cell.BasicLSTMCell(
                    dimension,
                    state_is_tuple=True),
                output_keep_prob=1 - self.lstm_dropout)],
            state_is_tuple=True)
