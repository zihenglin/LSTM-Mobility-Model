class AbstractLossFunction(object):
    """Compute the loss function based on
    maximum likelihood using the neural network
    output and the target sequence.
    """

    def __init__(self,
                 tensors,
                 mixture_density_builder):
        """
        Args:
            tensors(TensorBuilder): a TensorBuilder that holds
                all tensors for the model.
            mixture_density_builder(AbstractMixtureDensityBuilder):
                a AbstractMixtureDensityBuilder object that
                transforms lstm outputs into distribution
                parameters.
        """
        self.tensors = tensors
        self.mixture_density_builder = mixture_density_builder

    def get_loss(self):
        """Compute the total loss from lstm output and target
        sequences that can be optimized using optimizer.
        """
        raise NotImplementedError

    def _get_mask(self):
        """Get sequence masks that are used to mask loss that
        should be included for padded part of sequences.
        """
        raise NotImplementedError

    def _get_spatial_loss(self, spatial_distributions):
        """Get loss due to spatial distribution compared to
        target spatial sequences.
        """
        raise NotImplementedError

    def _get_temporal_loss(self, temporal_distributions):
        """Get loss due to temporal distribution compared to
        target temporal sequences.
        """
        raise NotImplementedError

    def _get_categorical_loss(self, categorical_distributions):
        """Get loss due to categorical distribution compared to
        target categorical sequences.
        """
        raise NotImplementedError
