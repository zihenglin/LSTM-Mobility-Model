import logging

import numpy as np
import tensorflow as tf


class Trainer(object):
    """Neural network trainer with custom train()
    function. The trainer minimize self.loss using
    self.optimizer.
    """

    def _random_sample_feed_dictionary(self,
                                       placeholders_dict,
                                       input_values_dict,
                                       batch_size):
        """Build self.feed_dict (a feed dictionary) for neural
        network training purpose. The keys are tf.placeholder
        and the values are np.array.
        Args:
            placeholders_dict(str -> tf.placeholder): a dictionary
                of placeholders with keys being placeholder names
                and values being the corresponding placeholders.
            input_values_dict(str -> tf.placeholder): a dictionary
                of input values with keys being input value names
                and values being the corresponding input values.
        """
        data_length = list(input_values_dict.values())[0].shape[0]
        rand_ind = np.random.choice(data_length,
                                    batch_size,
                                    replace=True)

        feed_dict = {}
        for key in input_values_dict:
            assert key in placeholders_dict
            feed_dict[placeholders_dict[key]] = input_values_dict[key][rand_ind]

        return feed_dict

    def train(self,
              optimizer,
              loss,
              placeholders_dict,
              input_values_dict,
              batch_size,
              epochs,
              tensorflow_session,
              verbose=True,
              logging_per=None):
        """Train the model given placeholders, input values,
        and other parameters.
        Args:
            optimizer(tf.train.Optimizer): an optimizer that minimize
                the loss when training.
            loss(tf.tensor): the compiled loss of the model that it can
                 be optimized by optimizer.
            placeholders_dict(str -> tf.placeholder): a dictionary
                of placeholders with keys being placeholder names
                and values being the corresponding placeholders.
            input_values_dict(str -> tf.placeholder): a dictionary
                of input values with keys being input value names
                and values being the corresponding input values.
            batch_size(int): the training batch size.
            epochs(int): the training epochs.
            tensorflow_session(tf.session): tensorflow session under
                which the variables lives.
            verbose(boolean): whether to log training loss information
                or not.
            logging_per(int): log training per unit of epoch. Default
                value is max(epochs / 10, 1).
        """

        if logging_per is None:
            logging_per = np.max([epochs / 10, 1])

        # Initialize all tensorflow variables
        tensorflow_session.run(tf.global_variables_initializer())

        total_loss = 0
        for e in range(epochs):

            feed_dict = self._random_sample_feed_dictionary(
                placeholders_dict,
                input_values_dict,
                batch_size)
            _, loss_value = tensorflow_session.run(
                [optimizer, loss],
                feed_dict=feed_dict)

            total_loss += np.array(loss_value)
            if verbose and (e + 1) % logging_per == 0:
                logging.info("Epoch: " + str(e + 1) + " Loss: " + str(total_loss /
                                                                      logging_per))
                total_loss = 0
