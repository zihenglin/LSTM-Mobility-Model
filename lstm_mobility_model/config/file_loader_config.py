from enum import Enum


class _SpatialColumnsLatLng(Enum):
    __order__ = ('lat '
                 'lon ')
    lat = 0
    lon = 1


class _SpatialColumnsDistHomeWork(Enum):
    __order__ = ('dist_to_home '
                 'dist_to_work ')
    dist_to_home = 0
    dist_to_work = 1


class _TemporalColumns(Enum):
    __order__ = ('start_hour_since_day '
                 'duration ')
    start_hour_since_day = 0
    duration = 1


class _ContextColumns(Enum):
    __order__ = ('start_dayofweek '
                 'start_week '
                 'start_date_id '
                 'end_dayofweek '
                 'end_week '
                 'end_date_id ')
    start_dayofweek = 0
    start_week = 1
    start_date_id = 2
    end_dayofweek = 3
    end_week = 4
    end_date_id = 5


class _CategoricalColumns(Enum):
    __order__ = ('is_home '
                 'is_work '
                 'is_other ')
    is_home = 0
    is_work = 1
    is_other = 2


class _ObservationFlags(Enum):
    __order__ = ('is_observed ')
    is_observed = 0


class Features(Enum):
    lat = _SpatialColumnsLatLng.lat.name
    lon = _SpatialColumnsLatLng.lon.name
    start_hour_since_day = _TemporalColumns.start_hour_since_day.name
    duration = _TemporalColumns.duration.name
    categorical_features = [_CategoricalColumns.is_home.name,
                            _CategoricalColumns.is_work.name,
                            _CategoricalColumns.is_other.name]
    contex_features = [_ContextColumns.start_dayofweek.name]
    initial_activity_type_input = 'initial_activity_type_input'
    mask = 'mask'


class OptionalFeatures(Enum):
    is_observed = _ObservationFlags.is_observed.name
    start_date_id = _ContextColumns.start_date_id.name
    dist_to_home = _SpatialColumnsDistHomeWork.dist_to_home.name
    dist_to_work = _SpatialColumnsDistHomeWork.dist_to_work.name
    is_home = _CategoricalColumns.is_home.name
    is_work = _CategoricalColumns.is_work.name
    is_other = _CategoricalColumns.is_other.name

    start_dayofweek = 'start_dayofweek'
    start_week = 'start_week'
    end_dayofweek = 'end_dayofweek'
    end_week = 'end_week'
    end_date_id = 'end_date_id'
    location_category = 'location_category'
    initial_location_category_input = 'initial_location_category_input'


class RequiredColumns(Enum):
    __order__ = ('lat '
                 'lon '
                 'start_hour_since_day '
                 'duration '
                 'is_home '
                 'is_work '
                 'is_other '
                 'start_dayofweek '
                 'start_week '
                 'end_dayofweek '
                 'end_week '
                 'location_category ')

    lat = _SpatialColumnsLatLng.lat.name
    lon = _SpatialColumnsLatLng.lon.name
    start_hour_since_day = _TemporalColumns.start_hour_since_day.name
    duration = _TemporalColumns.duration.name
    is_home = _CategoricalColumns.is_home.name
    is_work = _CategoricalColumns.is_work.name
    is_other = _CategoricalColumns.is_other.name
    start_dayofweek = _ContextColumns.start_dayofweek.name
    start_week = _ContextColumns.start_week.name
    end_dayofweek = _ContextColumns.end_dayofweek.name
    end_week = _ContextColumns.end_week.name
    location_category = 'location_category'
