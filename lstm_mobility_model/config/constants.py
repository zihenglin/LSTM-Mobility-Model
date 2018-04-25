import pytz

from .file_loader_config import Features


class Constants(object):

    # Location data filtering box
    BAY_AREA_LNG_MAX = -121.325055
    BAY_AREA_LNG_MIN = -123.029329
    BAY_AREA_LAT_MAX = 38.198519
    BAY_AREA_LAT_MIN = 37.097283

    # Longitude and latitude scaling
    LAT_CENTER = 37.6149675
    LON_CENTER = -122.1946135
    LON_SCALE = LAT_SCALE = 0.25

    # Time and duration scaling
    DURATION_MAX = 24.0
    START_TIME_MAX = 24.0

    # Activities
    MINIMUM_ACTIVITIES = 7
    MINIMUM_DAILY_ACTIVITIES = 2
    MAX_ACTIVITY_LEN = 8
    ACTIVITY_PADDING_LENGTH = MAX_ACTIVITY_LEN

    # IO Dimensions
    INPUT_LENGTH = MAX_ACTIVITY_LEN
    INTPUT_DIMENSION = 8
    NUMBER_OF_CATEGORIES = len(Features.categorical_features.value)


class DataProcessingConstants(object):

    # Location data filtering box
    BAY_AREA_LNG_MAX = -120
    BAY_AREA_LNG_MIN = -124
    BAY_AREA_LAT_MAX = 40
    BAY_AREA_LAT_MIN = 36

    # Data scaling
    LAT_CENTER = 37.6149675
    LON_CENTER = -122.1946135
    LON_SCALE = LAT_SCALE = 0.25

    # MAX_DIST_TRAVELED = 0.6
    DURATION_MAX = 24.0
    START_TIME_MAX = 24.0

    MINIMUM_ACTIVITIES = 7
    MINIMUM_DAILY_ACTIVITIES = 2
    MAX_ACTIVITY_LEN = 8
    ACTIVITY_PADDING_LENGTH = MAX_ACTIVITY_LEN

    TRAVEL_TIME_MAX = 1.0  # hour
    # TRAVEL_DIST_MAX = 0.22  # 100 km

    # Sampling
    ACTIVITY_MINIMUM_DURATION = 5. / 60

    DIST_HOME_WORK_MAX = 40000
    TZ = pytz.timezone("US/Pacific")
