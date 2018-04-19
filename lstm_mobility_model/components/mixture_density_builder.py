from lstm_mobility_model.config import Constants


class AbstractMixtureDensityBuilder(object):
    """It transforms the output from neural network
    into a set of parameters for probability mixtures
    functions.
    """
    DEFAULT_NUMBER_OF_MIXTURES = 10

    def __init__(self, number_of_mixtures=None):
        """
        Args:
            number_of_mixtures(int): number of mixture components
                in the output mixture distributions. The default
                value is DEFAULT_NUMBER_OF_MIXTURES.
        """
        self.number_of_mixtures = AbstractMixtureDensityBuilder.DEFAULT_NUMBER_OF_MIXTURES \
            if number_of_mixtures is None \
            else number_of_mixtures

    def get_mixture_density_output(self,
                                   neural_network_output):
        raise NotImplementedError

    def sample_from_density_output(self,
                                   neural_network_output):
        raise NotImplementedError

    def _sample_categorical(self,
                            neural_network_output):
        raise NotImplementedError

    def _sample_temporal(self,
                         neural_network_output):
        raise NotImplementedError

    def _sample_spatial(self,
                        neural_network_output):
        raise NotImplementedError
