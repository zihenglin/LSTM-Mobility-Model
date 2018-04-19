from lstm_mobility_model.components import TensorBuilder
from lstm_mobility_model.config import (Features,
                                        OptionalFeatures,
                                        Constants)


class TwoLayerTensorBuilder(TensorBuilder):
    DEFAULT_PARAMETERS_PER_MIXTURE = 10

    def __init__(self,
                 lstm_units,
                 number_of_mixtures,
                 lstm_dropout=None,
                 batch_size=None,
                 parameters_per_mixture=None,
                 context_dimensions=1):
        """
        Args:
            lstm_units (int): number of lstm units.
            lstm_dropout (float): the dropout rates of lstm units.
            number_of_mixtures (int): number of mixtures in the
                Gaussian distribution.
            batch_size (int): batch size.
        """
        TensorBuilder.__init__(self,
                               lstm_dropout=lstm_dropout,
                               batch_size=batch_size)

        self.lstm_units = lstm_units
        self.number_of_mixtures = number_of_mixtures
        self.layer_2_output_parameters = \
            self.number_of_mixtures * TwoLayerTensorBuilder.DEFAULT_PARAMETERS_PER_MIXTURE
        self.context_dimensions = context_dimensions

        self.build_placeholders(self._get_placeholder_dimensions())
        self.build_trainable_variables(self._get_trainable_tensor_dimensions())
        self.build_lstm_layers(self._get_lstm_dimensions())

    def _get_placeholder_dimensions(self):
        """Get dimension of placeholders as dictionaries.
        Returns:
            dict(str -> list(int)): tensor names and dimensions
                map as dictionaries.
        """
        return {Features.contex_features.name: [None,
                                                Constants.INPUT_LENGTH,
                                                self.context_dimensions],
                Features.lat.name: [None,
                                    Constants.INPUT_LENGTH,
                                    1],
                Features.lon.name: [None,
                                    Constants.INPUT_LENGTH,
                                    1],
                Features.start_hour_since_day.name: [None,
                                                     Constants.INPUT_LENGTH,
                                                     1],
                Features.duration.name: [None,
                                         Constants.INPUT_LENGTH,
                                         1],
                Features.categorical_features.name: [None,
                                                     Constants.INPUT_LENGTH,
                                                     1],
                Features.contex_features.name: [None,
                                                Constants.INPUT_LENGTH,
                                                1],
                Features.mask.name: [None, Constants.INPUT_LENGTH],
                Features.initial_activity_type_input.name: [None, 1],
                OptionalFeatures.is_observed.name: [None,
                                                    Constants.INPUT_LENGTH,
                                                    1]}

    def _get_trainable_tensor_dimensions(self):
        """Get trainable tensor dimensions.
        Returns:
            dict(str -> list(int)): tensor names and dimensions
                map as dictionaries.
        """
        return {'output_embedding_layer_1': [self.lstm_units,
                                             Constants.NUMBER_OF_CATEGORIES],
                'output_bias_layer_1': [1, Constants.NUMBER_OF_CATEGORIES],
                'output_embedding_layer_2': [self.lstm_units,
                                             self.layer_2_output_parameters],
                'output_bias_layer_2': [1, self.layer_2_output_parameters]}

    def _get_lstm_dimensions(self):
        """Get LSTM dimensions.
        Returns:
            dict(str -> list(int)): tensor names and dimensions
                map as dictionaries.
        """
        return {'lstm_layer_1': self.lstm_units,
                'lstm_layer_2': self.lstm_units}
