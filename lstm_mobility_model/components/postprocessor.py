import pandas as pd
import numpy as np

from lstm_mobility_model.config import (Features,
                                        OptionalFeatures,
                                        DataProcessingConstants)


class LatLonPostProcessor(object):
    def __init__(self):
        self.columns = [Features.start_hour_since_day.name,
                        Features.duration.name,
                        Features.lat.name,
                        Features.lon.name,
                        Features.categorical_features.name]

    def post_processing(self,
                        generated_sequences,
                        observed_sequences,
                        context_sequences,
                        masks,
                        observation_indices,
                        trim_mode=None):
        """The post processing steps scales each generated features
        back to its original scale. The default feature order is as
        the following:
            self.generated_activity_start_time,
            self.generated_activity_duration,
            self.generated_activity_lat,
            self.generated_activity_lon,
            self.generated_activity_type

        Args:
            trim_mode(str): string that specifies how the activity sequences
                are trimmed. Choices are:
                    None
                    "next_activity"
                    "by_time"
                    "partially_observed"
        """

        if trim_mode == "next_activity":
            sequence_length = int(np.sum(masks))
            generated_sequences = generated_sequences[:sequence_length]

        elif trim_mode is None or trim_mode == "by_time":
            last_activity_index = np.where(generated_sequences[:, 0] +
                                           generated_sequences[:, 1] > 24 /
                                           DataProcessingConstants.DURATION_MAX)
            if len(last_activity_index[0]) > 0:
                generated_sequences = generated_sequences[:last_activity_index[0][0] + 1]

        elif trim_mode == 'partially_observed':

            generated_sequences = observation_indices * observed_sequences + \
                (1 - observation_indices) * generated_sequences

            last_activity_index = np.where(generated_sequences[:, 0] +
                                           generated_sequences[:, 1] > 24 /
                                           DataProcessingConstants.DURATION_MAX)

            if len(last_activity_index[0]) > 0:
                generated_sequences = generated_sequences[:last_activity_index[0][0] + 1]

        # Scale the units back for output
        # Start time
        generated_sequences[:, 0] *= DataProcessingConstants.START_TIME_MAX

        # Duration
        generated_sequences[:, 1] *= DataProcessingConstants.DURATION_MAX

        # Latitude
        generated_sequences[:, 2] *= DataProcessingConstants.LAT_SCALE
        generated_sequences[:, 2] += DataProcessingConstants.LAT_CENTER

        # Longitude
        generated_sequences[:, 3] *= DataProcessingConstants.LON_SCALE
        generated_sequences[:, 3] += DataProcessingConstants.LON_CENTER

        return pd.DataFrame(data=generated_sequences, columns=self.columns)

    def get_observations(self,
                         input_values_dict,
                         item_index):
        # Observed sequences
        observed_sequences = [input_values_dict[column_name][item_index]
                              for column_name in self.columns]
        observed_sequences = np.concatenate(observed_sequences, axis=-1)

        # Observation index
        observation_indices = input_values_dict[
            OptionalFeatures.is_observed.name][item_index]

        return observed_sequences, \
            input_values_dict[Features.contex_features.name][item_index], \
            observation_indices


class CategoricalLocationPostProcessor(LatLonPostProcessor):
    def __init__(self, location_lat_lng_map):
        self.location_lat_lng_map = location_lat_lng_map
        self.columns = [Features.start_hour_since_day.name,
                        Features.duration.name,
                        OptionalFeatures.location_category.name,
                        Features.categorical_features.name]

    def post_processing(self,
                        generated_sequences,
                        observed_sequences,
                        context_sequences,
                        masks,
                        observation_indices,
                        trim_mode=None):
        """The post processing steps scales each generated features
        back to its original scale. The default feature order is as
        the following:
            self.generated_activity_start_time,
            self.generated_activity_duration,
            self.generated_activity_location,
            self.generated_activity_type
        Args:
            trim_mode(str): string that specifies how the activity sequences
                are trimmed. Choices are:
                    None
                    "next_activity"
                    "by_time"
                    "partially_observed"
        """

        if trim_mode == "next_activity":
            sequence_length = int(np.sum(masks))
            generated_sequences = generated_sequences[:sequence_length]

        elif trim_mode == 'partially_observed':

            generated_sequences = observation_indices * observed_sequences + \
                (1 - observation_indices) * generated_sequences

            last_activity_index = np.where(generated_sequences[:, 0] +
                                           generated_sequences[:, 1] > 24 /
                                           DataProcessingConstants.DURATION_MAX)

            if len(last_activity_index[0]) > 0:
                generated_sequences = generated_sequences[:last_activity_index[0][0] + 1]

        elif trim_mode is None or trim_mode == "by_time":
            last_activity_index = np.where(generated_sequences[:, 0] +
                                           generated_sequences[:, 1] > 24 /
                                           DataProcessingConstants.DURATION_MAX)
            if len(last_activity_index[0]) > 0:
                generated_sequences = generated_sequences[:last_activity_index[0][0] + 1]

        # Trim context sequences
        context_sequences = context_sequences[:len(generated_sequences)]

        # Scale the units back for output
        # Start time and duration
        generated_sequences[:, 0] *= DataProcessingConstants.START_TIME_MAX
        generated_sequences[:, 1] *= DataProcessingConstants.DURATION_MAX
        generated_lat, generated_lng = self._get_lat_lng_from_persona(
            generated_sequences[:, 2],
            self.location_lat_lng_map)

        # Column names
        columns = [Features.start_hour_since_day.name,
                   Features.duration.name,
                   OptionalFeatures.location_category.name,
                   Features.categorical_features.name,
                   Features.lat.name,
                   Features.lon.name]
        for i in range(context_sequences.shape[1]):
            columns.append(Features.contex_features.name + "_{}".format(i))

        return pd.DataFrame(data=np.hstack([generated_sequences, generated_lat, generated_lng, context_sequences]),
                            columns=columns)

    def _get_lat_lng_from_persona(self,
                                  generated_location_categories,
                                  location_lat_lng_map):
        """Convert generaterated location categories, a list of int,
        into a list of latitude and a list of longitude based on habitat
        in the given location_lat_lng_map.
        Args:
             generated_location_categories(list(int)): the generated
                location categories as a list of int.
             location_lat_lng_map(dict(int -> tuple())): map from location
                categories to lat lng tuple.
        Returns:
            (list(float)): generated latitude
            (list(float)): generated longitude
        """
        lat_lng_list = np.array([list(location_lat_lng_map[category])
                                 for category in generated_location_categories])
        return lat_lng_list[:, 0][:, np.newaxis], lat_lng_list[:, 1][:, np.newaxis]
