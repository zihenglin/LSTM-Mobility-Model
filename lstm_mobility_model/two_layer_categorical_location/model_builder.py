import os
import logging

import tensorflow as tf
import numpy as np

from lstm_mobility_model.config import (Constants,
                                        Features,
                                        OptionalFeatures)
from lstm_mobility_model.components.postprocessor import (
    CategoricalLocationPostProcessor)
from lstm_mobility_model.components import (AbstractLstmModelBuilder,
                                            FeatureBuilder,
                                            Trainer,
                                            Generator)
from lstm_mobility_model.two_layer_latlng_location import TwoLayerLstmModelBuilder
from lstm_mobility_model.load import (LstmInputLoader,
                                      DataPreprocessor)
from .lstm_builder import (TwoLayerCategoricalLocationLSTMGenerating,
                           TwoLayerCategoricalLocationLSTMTraining)
from .loss_function import TwoLayerCategoricalLocationLossFunction
from .mixture_density_builder import TwoLayerCategoricalLocationMixtureDensityBuilder
from .tensor_builder import TwoLayerTensorCategoricalLocationBuilder


class _ModelMode(object):
    TRAINING = 'TRAINING'
    GENERATING = 'GENERATING'


class TwoLayerLstmCategoricalLocationModelBuilder(
        TwoLayerLstmModelBuilder):

    DEFAULT_FEATURE_LIST = [Features.start_hour_since_day.name,
                            Features.duration.name,
                            Features.categorical_features.name,
                            Features.contex_features.name,
                            Features.initial_activity_type_input.name,
                            Features.mask.name,
                            OptionalFeatures.location_category.name,
                            OptionalFeatures.initial_location_category_input.name]

    def __init__(self,
                 location_lat_lng_map,
                 model_path=None,
                 model_name=None,
                 learning_rate=None,
                 training_epochs=None,
                 lstm_dropout=None,
                 lstm_units=None,
                 number_of_mixtures=None,
                 sampling_bias=None):
        TwoLayerLstmModelBuilder.__init__(
            self,
            model_path=model_path,
            model_name=model_name,
            learning_rate=learning_rate,
            training_epochs=training_epochs,
            lstm_dropout=lstm_dropout,
            lstm_units=lstm_units,
            number_of_mixtures=number_of_mixtures,
            sampling_bias=sampling_bias)

        # Model peripherals
        self.feature_builder = FeatureBuilder()
        self.mixture_density_builder = TwoLayerCategoricalLocationMixtureDensityBuilder()
        self.trainer = Trainer()
        self.generator = Generator(
            post_processor=CategoricalLocationPostProcessor(
                location_lat_lng_map=location_lat_lng_map))
        self.n_location_categories = len(location_lat_lng_map)

    def train(self,
              traces_dict,
              batch_size=100,
              preprocess_data=True,
              logging_per=None):
        """Train model with dictionary of traces by day.
        This is to be consistant with other model interfaces.
        Args:
            traces_dict(object->pd.DataFrame): the dataframe by date.
            batch_size(int): training batch size.
            preprocess_data(bool): if preprocessing and scaleing data is needed.
            logging_per(int): print loss per logging_per interations.
        """
        # Preprocess data if required. Skip will speed up performance.
        if preprocess_data:
            data_preprocessor = DataPreprocessor()
            traces_dict = data_preprocessor.preprocess_traces_df_dict(
                traces_dict,
                categorical_location=True)
        data_loader = LstmInputLoader()
        lstm_input_values_dict = data_loader.get_lstm_features_from_traces_dict(
            traces_dict=traces_dict,
            features=TwoLayerLstmCategoricalLocationModelBuilder.DEFAULT_FEATURE_LIST)

        self.train_with_formated_data(lstm_input_values_dict,
                                      batch_size=batch_size,
                                      epochs=self.training_epochs,
                                      logging_per=logging_per)

        self.mode = _ModelMode.TRAINING

    def train_with_formated_data(self,
                                 input_values_dict,
                                 batch_size,
                                 epochs,
                                 logging_per=None):
        """Train model with formated data, input_values_dict, as dictionary of
        np.arrays.
        """
        # Build training model graph
        tf.reset_default_graph()
        self.session = tf.Session()
        self.tensors = TwoLayerTensorCategoricalLocationBuilder(
            n_location_categories=self.n_location_categories,
            lstm_units=self.lstm_units,
            lstm_dropout=self.lstm_dropout,
            number_of_mixtures=self.number_of_mixtures,
            batch_size=batch_size)
        self._build_training_model(self.tensors)
        placeholders_dict = self.tensors.placeholders

        # Initialize variables
        self.session.run(tf.global_variables_initializer())

        # Train
        self.trainer.train(
            optimizer=[self.optimizer_layer_1, self.optimizer_layer_2],
            loss=[self.loss_layer_1, self.loss_layer_2],
            placeholders_dict=placeholders_dict,
            input_values_dict=input_values_dict,
            batch_size=batch_size,
            epochs=epochs,
            tensorflow_session=self.session,
            logging_per=logging_per)

        # Save model
        self._save_model(tensorflow_session=self.session)

    def _build_generating_model(self, tensors):
        """Build training model on a tensorflow graph.
        """
        # Generating model
        lstm_generating = \
            TwoLayerCategoricalLocationLSTMGenerating(
                tensors=tensors,
                mixture_density_output=self.mixture_density_builder,
                feature_builder=self.feature_builder,
                sampling_bias=self.sampling_bias)
        lstm_generating.build_lstm()

        # Build sequence generation
        self.sequence_generating_tensors = lstm_generating.get_sequence_tensors()

    def _build_training_model(self, tensors):
        """Build training model on a tensorflow graph.
        """
        # Training model
        lstm_training = \
            TwoLayerCategoricalLocationLSTMTraining(
                tensors=tensors,
                mixture_density_output=self.mixture_density_builder,
                feature_builder=self.feature_builder)
        lstm_training.build_lstm()

        # Build loss
        layer_1_output_socres,\
            layer_2_output_socres = lstm_training.get_sequence_tensors()
        loss_function = TwoLayerCategoricalLocationLossFunction(
            tensors=tensors,
            mixture_density_builder=self.mixture_density_builder)

        self.loss_layer_1, self.loss_layer_2 = loss_function.get_loss(
            self.n_location_categories,
            layer_1_output_socres,
            layer_2_output_socres)
        self.optimizer_layer_1 = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate,
            beta1=0.9,
            beta2=0.85).minimize(self.loss_layer_1)
        self.optimizer_layer_2 = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate,
            beta1=0.8,
            beta2=0.9).minimize(self.loss_layer_2)

    def generate(self,
                 traces_dict,
                 cut_time=None,
                 method=None,
                 preprocess_data=True):
        """This is the interface for validator that it should generate sequences
        for the dates in traces_dict.
        Args:
            traces_dict(object->pd.DataFrame): the dataframe by date.
            method(str): the name of the method. It can be
                'next_activity' or
                'complete' or
                'partially_observed'.
        Returns:
            (object->pd.DataFrame): the generated dataframe by date.
        """
        # Default genration method
        method = 'next_activity' if method is None else method

        # Preprocess data if required. Skip will speed up performance.
        if preprocess_data:
            data_preprocessor = DataPreprocessor()
            traces_dict = data_preprocessor.preprocess_traces_df_dict(
                traces_dict,
                categorical_location=True)

        # Get keys and values from dataframe_dict
        data_loader = LstmInputLoader()
        lstm_input_values = data_loader.get_lstm_features_from_partial_traces_dict(
            traces_dict,
            cut_time,
            features=TwoLayerLstmCategoricalLocationModelBuilder.DEFAULT_FEATURE_LIST)

        # Generate sequences
        generated_sequence_keys = traces_dict.keys()
        if method == 'complete':
            generated_sequence_dataframe = self.generate_complete_sequences(
                lstm_input_values)
        elif method == 'partially_observed':
            generated_sequence_dataframe = self.generate_partial_sequences(
                lstm_input_values)
        else:
            logging.warning('Invalid generate method. Return None.')
            return

        # Output back as dict
        return dict(zip(generated_sequence_keys,
                        generated_sequence_dataframe))
