import os
import logging

import tensorflow as tf
import numpy as np

from lstm_mobility_model.config import (Constants,
                                        Features,
                                        OptionalFeatures)
from lstm_mobility_model.components import (AbstractLstmModelBuilder,
                                            FeatureBuilder,
                                            Trainer,
                                            Generator)
from lstm_mobility_model.load import (LstmInputLoader,
                                      DataPreprocessor)
from .lstm_builder import (TwoLayerLSTMGenerating,
                           TwoLayerLSTMTraining)
from .loss_function import TwoLayerLossFunction
from .mixture_density_builder import TwoLayerMixtureDensityBuilder
from .tensor_builder import TwoLayerTensorBuilder


class _ModelMode(object):
    TRAINING = 'TRAINING'
    GENERATING = 'GENERATING'


class TwoLayerLstmModelBuilder(AbstractLstmModelBuilder):

    DEFAULT_MODEL_PATH = 'lstm_model_weights/'
    DEFAULT_MODEL_NAME = 'two_layer_spatial_temporal_model'
    DEFAULT_LEARNING_RATE = 0.1
    DEFAULT_LSTM_DROPOUT = 0.00
    DEFAULT_LSTM_UNITS = 8
    DEFAULT_SAMPLING_BIAS = 0.0
    DEFAULT_TRAINING_EPOCHS = 1000

    def __init__(self,
                 model_path=None,
                 model_name=None,
                 learning_rate=None,
                 training_epochs=None,
                 lstm_dropout=None,
                 lstm_units=None,
                 number_of_mixtures=None,
                 sampling_bias=None):
        """
        Args:
            model_path(str): the path that the model weights should be
                stored.
            model_name(str): the file name of the model weights to be
                stored and loaded.
        """

        # Model peripherals
        self.feature_builder = FeatureBuilder()
        self.mixture_density_builder = TwoLayerMixtureDensityBuilder()
        self.trainer = Trainer()
        self.generator = Generator()

        # Compiled tensors
        self.loss = None
        self.optimizer = None
        self.sequence_generating_tensors = None
        self.tensors = None

        # Model mode
        self.mode = None
        self.session = None

        # Check and create model path
        self.model_path = TwoLayerLstmModelBuilder.DEFAULT_MODEL_PATH \
            if model_path is None \
            else model_path
        self.model_name = TwoLayerLstmModelBuilder.DEFAULT_MODEL_NAME \
            if model_name is None \
            else model_name
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        self.model_file_path = os.path.join(self.model_path,
                                            self.model_name)

        # Tunings
        self.learning_rate = \
            TwoLayerLstmModelBuilder.DEFAULT_LEARNING_RATE \
            if learning_rate is None \
            else learning_rate

        self.training_epochs = \
            TwoLayerLstmModelBuilder.DEFAULT_TRAINING_EPOCHS \
            if training_epochs is None \
            else training_epochs

        self.lstm_dropout = \
            TwoLayerLstmModelBuilder.DEFAULT_LSTM_DROPOUT \
            if lstm_dropout is None \
            else lstm_dropout

        self.lstm_units = TwoLayerLstmModelBuilder.DEFAULT_LSTM_UNITS \
            if lstm_units is None \
            else lstm_units

        self.number_of_mixtures = \
            Constants.DEFAULT_NUMBER_OF_MIXTURES\
            if number_of_mixtures is None \
            else number_of_mixtures

        self.sampling_bias = \
            TwoLayerLstmModelBuilder.DEFAULT_SAMPLING_BIAS\
            if sampling_bias is None \
            else sampling_bias

    def _build_generating_model(self, tensors):
        """Build training model on a tensorflow graph.
        """
        # Generating model
        lstm_generating = \
            TwoLayerLSTMGenerating(tensors=tensors,
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
            TwoLayerLSTMTraining(tensors,
                                 self.mixture_density_builder,
                                 self.feature_builder)
        lstm_training.build_lstm()

        # Build loss
        layer_1_output_socres,\
            layer_2_output_socres = lstm_training.get_sequence_tensors()
        loss_function = TwoLayerLossFunction(
            tensors=tensors,
            mixture_density_builder=self.mixture_density_builder)

        self.loss_layer_1, self.loss_layer_2 = loss_function.get_loss(layer_1_output_socres,
                                                                      layer_2_output_socres)
        self.optimizer_layer_1 = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate,
            beta1=0.9,
            beta2=0.85).minimize(self.loss_layer_1)
        self.optimizer_layer_2 = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate,
            beta1=0.8,
            beta2=0.9).minimize(self.loss_layer_2)

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
        self.tensors = TwoLayerTensorBuilder(
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

    def train(self,
              traces_dict,
              batch_size=100,
              preprocess_data=True,
              logging_per=None):
        """Train model with dictionary of traces by day.
        This is to be consistant with other model interfaces.
        """
        # Preprocess data if required. Skip will speed up performance.
        if preprocess_data:
            data_preprocessor = DataPreprocessor()
            traces_dict = data_preprocessor.preprocess_traces_df_dict(traces_dict)

        data_loader = LstmInputLoader()
        lstm_input_values_dict = data_loader.get_lstm_features_from_traces_dict(traces_dict)

        self.train_with_formated_data(lstm_input_values_dict,
                                      batch_size=batch_size,
                                      epochs=self.training_epochs,
                                      logging_per=logging_per)

        self.mode = _ModelMode.TRAINING

    def _generate_sequences(self,
                            input_values_dict,
                            trim_mode=None):
        """Generate sequence generation function.
        """

        # Build and load model if model has not been
        # built or loaded.
        if self.mode != _ModelMode.GENERATING:
            self._build_generating_model(self.tensors)

        self.mode = _ModelMode.GENERATING

        return self.generator.generate(
            self.sequence_generating_tensors,
            self.tensors.placeholders,
            input_values_dict,
            tensorflow_session=self.session,
            trim_mode=trim_mode)

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
            traces_dict = data_preprocessor.filter_traces_dict(traces_dict)
            traces_dict = data_preprocessor.preprocess_traces_df_dict(traces_dict)

        # Get keys and values from dataframe_dict
        data_loader = LstmInputLoader()
        lstm_input_values = data_loader.get_lstm_features_from_partial_traces_dict(
            traces_dict,
            cut_time)

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

    def generate_complete_sequences(self,
                                    input_values_dict):
        """Generate the complete sequences starting from the beginning
        of the day.
        """
        data_size = len(list(input_values_dict.values())[0])
        input_values_dict[OptionalFeatures.is_observed.name] = np.zeros((
            data_size,
            Constants.INPUT_LENGTH,
            1))

        return self._generate_sequences(input_values_dict)

    def generate_partial_sequences(self,
                                   input_values_dict):
        """Generate partially observed sequences. The observation
        in indices are in input_values_dict with key of
        OptionalFeatures.is_observed.name.
        """
        assert OptionalFeatures.is_observed.name in input_values_dict
        generated_sequences = self._generate_sequences(input_values_dict,
                                                       trim_mode='partially_observed')
        return generated_sequences

    def _save_model(self, tensorflow_session):
        """Save model weights to self.model_file_path given
        a tensorflow session.
        """
        saver = tf.train.Saver()
        save_path = saver.save(sess=tensorflow_session,
                               save_path=self.model_file_path,
                               write_meta_graph=False,
                               write_state=False)
        logging.info("Saved model to {}".format(save_path))

    def _load_model(self, tensorflow_session):
        """Load model weights from self.model_file_path given
        a tensorflow session.
        """
        saver = tf.train.Saver()
        saver.restore(sess=tensorflow_session,
                      save_path=self.model_file_path)
        logging.info("Resported model from {}".format(self.model_file_path))
