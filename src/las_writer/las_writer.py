import pathlib

import numpy as np
import pandas as pd

import laspy

MAX_LONG = np.iinfo(np.int32).max


class LasHeaderOffsetScaleDefiner(object):
    def __init__(self, points, x_column_name="x_lidar", y_column_name="y_lidar", z_column_name="z_lidar", amplitude=MAX_LONG):
        self.amplitude = amplitude

        self.min_x, self.max_x = self.get_min_max(points[x_column_name])
        self.min_y, self.max_y = self.get_min_max(points[y_column_name])
        self.min_z, self.max_z = self.get_min_max(points[z_column_name])

        self.offset_x = self.compute_offset(self.min_x, self.max_x)
        self.scale_x = self.compute_scale(self.min_x, self.max_x, self.offset_x)

        self.offset_y = self.compute_offset(self.min_y, self.max_y)
        self.scale_y = self.compute_scale(self.min_y, self.max_y, self.offset_y)

        self.offset_z = self.compute_offset(self.min_z, self.max_z)
        self.scale_z = self.compute_scale(self.min_z, self.max_z, self.offset_z)

    def get_min_max(self, column):
        return column.min(), column.max()

    @property
    def amplitude(self):
        return self._amplitude

    @amplitude.setter
    def amplitude(self, new_amplitude):
        assert new_amplitude >= 0, "amplitude should be positive"
        assert new_amplitude <= MAX_LONG, "amplitude should be of type long... because we need to store our coordinates in a 4 bytes long"
        self._amplitude = new_amplitude

    @property
    def offset(self):
        return np.array([self.offset_x, self.offset_y, self.offset_z])

    @property
    def scale(self):
        return np.array([self.scale_x, self.scale_y, self.scale_z])

    def compute_offset(self, min_value, max_value):
        mean = (max_value - min_value) / 2
        offset = min_value + mean
        return offset

    def compute_scale(self, min_value, max_value, offset_column):
        min_value_with_offset = self.apply_offset(min_value, offset_column)
        max_value_with_offset = self.apply_offset(max_value, offset_column)
        scale = max(abs(min_value_with_offset), abs(max_value_with_offset)) / self.amplitude
        return scale

    def apply_offset(self, value, offset):
        return value - offset

    def get_coordinate_to_save_in_las(self, point):
        [x, y, z] = point
        assert x >= self.min_x
        assert x <= self.max_x
        assert y >= self.min_y
        assert y <= self.max_y
        assert z >= self.min_z
        assert z <= self.max_z

        scaled_x = self.apply_offset(x, self.offset_x) / self.scale_x
        scaled_y = self.apply_offset(y, self.offset_y) / self.scale_y
        scaled_z = self.apply_offset(z, self.offset_z) / self.scale_z

        return np.round(np.array([scaled_x, scaled_y, scaled_z]))

    def vectorized_get_coordinates_to_save_in_las(self, points, x_column_name="x_lidar", y_column_name="y_lidar", z_column_name="z_lidar"):

        points['new_x'] = (points[x_column_name] - self.offset_x) / self.scale_x
        points['new_y'] = (points[y_column_name] - self.offset_y) / self.scale_y
        points['new_z'] = (points[z_column_name] - self.offset_z) / self.scale_z

        return points

    def get_raw_data(self, scaled_coordinate):
        raise NotImplementedError()
        assert abs(scaled_coordinate) <= self.amplitude
        raw_coordinate = (scaled_coordinate * self.scale) + self.offset
        return raw_coordinate

    def estimate_precision_between_consecutive_saved_coordinates(self):
        raise NotImplementedError()
        coordinate_a = np.array([self.min_x, self.min_y, self.min_z])

        minimum_long_saved_in_las = self.get_coordinate_to_save_in_las(coordinate_a)
        next_long_saved_in_las = minimum_long_saved_in_las + 1

        raw_coordinate_a = self.get_raw_data(minimum_long_saved_in_las)
        raw_coordinate_b = self.get_raw_data(next_long_saved_in_las)

        difference_between_closest_raw_data = raw_coordinate_b - raw_coordinate_a
        # Compute the norm here

        return difference_between_closest_raw_data


class JakartoLasHeaderOffsetScaleDefiner(LasHeaderOffsetScaleDefiner):
    def __init__(self, points, x_column_name="x_lidar", y_column_name="y_lidar", z_column_name="z_lidar", amplitude=MAX_LONG):
        super().__init__(points, x_column_name=x_column_name, y_column_name=y_column_name, z_column_name=z_column_name, amplitude=amplitude)
        # TODO(tofull) uncomment the next check once the estimate_precision_between_consecutive_saved_coordinates is well implemented
        # precision_goal_in_m = 0.1e-3  # 1 tenth of millimeter
        # assert self.estimate_precision_between_consecutive_saved_coordinates() < precision_goal_in_m


class DataAggregator(object):
    def __init__(self, dtype):
        self.dtype = dtype

    def convert_points(self, points):
        # Dtype conversion
        points['X'] = points['X'].astype('<i4')
        points['Y'] = points['Y'].astype('<i4')
        points['Z'] = points['Z'].astype('<i4')

        points['intensity'] = points['intensity'].astype('<u2')

        points['gps_time'] = points['gps_time'].astype('<f8')

        # Inserting missing columns
        points['flag_byte'] = 0
        points['flag_byte'] = points['flag_byte'].astype('u1')

        points['raw_classification'] = 0
        points['raw_classification'] = points['raw_classification'].astype('u1')

        points['scan_angle_rank'] = 0
        points['scan_angle_rank'] = points['scan_angle_rank'].astype('u1')

        points['user_data'] = 0
        points['user_data'] = points['user_data'].astype('u1')

        points['pt_src_id'] = 0
        points['pt_src_id'] = points['pt_src_id'].astype('u1')

        # Reorder columns
        points = points[['X', 'Y', 'Z', 'intensity', 'flag_byte', "raw_classification", "scan_angle_rank", "user_data", "pt_src_id", 'gps_time']]

        # Convert pandas dataframe to numpy structured array
        points_to_numpy = points.to_records(index=False).view(dtype=[('point', [('X', '<i4'), ('Y', '<i4'), ('Z', '<i4'), ('intensity', '<u2'), ('flag_byte', 'u1'), ('raw_classification', 'u1'), ('scan_angle_rank', 'u1'), ('user_data', 'u1'), ('pt_src_id', 'u1'), ('gps_time', '<f8')])])
        points_to_numpy = np.asarray(points_to_numpy)
        points_to_numpy = points_to_numpy.astype([('point', [('X', '<i4'), ('Y', '<i4'), ('Z', '<i4'), ('intensity', '<u2'), ('flag_byte', 'u1'), ('raw_classification', 'u1'), ('scan_angle_rank', 'i1'), ('user_data', 'u1'), ('pt_src_id', '<u2'), ('gps_time', '<f8')])])

        return points_to_numpy


class JakartoLasFile(object):
    def __init__(self, output_filename, las_header_offset_scaler_instance):
        self.output_filename = output_filename

        template_filename = pathlib.Path(__file__).parent / "template.las"
        template_filename = template_filename.resolve()
        template_filename = template_filename.as_posix()

        self.template_file = laspy.file.File(template_filename, mode="r")

        points_from_template_file = self.template_file.get_points()
        dtype_from_template_file = points_from_template_file.dtype
        self.data_aggregator = DataAggregator(dtype_from_template_file)

        self.jakarto_header_offset_scale_definer = las_header_offset_scaler_instance

    def write_and_close(self, points):
        header_from_template_file = self.template_file.header
        self.output_file = laspy.file.File(self.output_filename, mode="w", header=header_from_template_file)

        offset = self.jakarto_header_offset_scale_definer.offset
        scale = self.jakarto_header_offset_scale_definer.scale
        self.output_file.header.scale = scale
        self.output_file.header.offset = offset

        self.data_aggregator.convert_points(points)

        self.output_file.points = self.data_aggregator.convert_points(points)

        self.output_file.close()
        self.template_file.close()

