import numpy as np

from lstm_mobility_model.config import (Features,
                                        OptionalFeatures)
from .postprocessor import LatLonPostProcessor


class Generator(object):
    """Sequence generator that generates entire sequences
    or generates sequences with partially observed input
    sequences.
    """

    def __init__(self, post_processor=None):
        """
        Args:
            post_processor(object): post processor object
                for projecting generated values into original
                space. Default is an LatLonPostProcessor object.
        """
        self.post_processor = LatLonPostProcessor() \
            if not post_processor \
            else post_processor

    def generate(self,
                 lstm_output,
                 placeholders_dict,
                 input_values_dict,
                 tensorflow_session,
                 trim_mode=None):
        """Generate sequences from trained LSTM model based on the
            intput sequences.
        Args:
            lstm_output(tf.tensor): unrolled lstm output tensor
                to be used for generating sequences.
            placeholders_dict(str -> tf.placeholder): a dictionary
                of placeholders with keys being placeholder names
                and values being the corresponding placeholders.
            input_values_dict(str -> tf.placeholder): a dictionary
                of input values with keys being input value names
                and values being the corresponding input values.
            epochs(int): the training epochs.
            tensorflow_session(tf.session): tensorflow session under
                which the variables lives.
            trim_mode(str): string that specifies how the activity sequences
                are trimmed. Choices are:
                    None
                    "next_activity"
                    "partially_observed"

        """
        data_size = len(list(input_values_dict.values())[0])

        output_sequences = []
        for i in range(data_size):
            feed_dict = self._get_feed_dictionary(placeholders_dict,
                                                  input_values_dict,
                                                  i)
            # Generated sequences
            generated_sequences = tensorflow_session.run(
                lstm_output,
                feed_dict=feed_dict)
            generated_sequences = np.squeeze(np.array(generated_sequences).astype(float))
            generated_sequences = np.transpose(generated_sequences, [1, 0])

            observed_sequences, context_sequences, observation_indices = \
                self.post_processor.get_observations(input_values_dict, i)

            output_sequences.append(
                self.post_processor.post_processing(
                    generated_sequences=generated_sequences,
                    observed_sequences=observed_sequences,
                    context_sequences=context_sequences,
                    masks=input_values_dict[Features.mask.name][i],
                    observation_indices=observation_indices,
                    trim_mode=trim_mode))
        return output_sequences

    def _get_feed_dictionary(self,
                             placeholders_dict,
                             input_values_dict,
                             item_index=None):
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
            item_index(int): index of the item number in input_values_dict
                that will be used for the output.
        """

        feed_dict = {}
        for key in input_values_dict:
            assert key in placeholders_dict
            if item_index is None:
                feed_dict[placeholders_dict[key]] = input_values_dict[key]
            else:
                feed_dict[placeholders_dict[key]] = input_values_dict[key][[item_index]]
        return feed_dict
