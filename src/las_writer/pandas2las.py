import pandas as pd
from las_writer.las_writer import JakartoLasHeaderOffsetScaleDefiner
from las_writer.las_writer import JakartoLasFile


def pandas2las(dataframe, output_filename):
    # Save a pandas dataframe to a las file
    # Input :
    # dataframe -> pandas dataframe indexed by gps_time with ['X', 'Y', 'Z', 'intensity', 'gps_time', 'x_lidar','y_lidar','z_lidar','x_gps','y_gps','z_gps','heading_imu','roll_imu','pitch_imu'] columns
    # output_filename -> string output filename

    dfsave = dataframe[['gps_time', 'intensity', 'X', 'Y', 'Z']]
    dfsave = dfsave.rename(columns={
        'X': 'x_lidar',
        'Y': 'y_lidar',
        'Z': 'z_lidar'
    }, copy=False)
    ptCloud = dfsave
    las_header_offset_scale_definer = JakartoLasHeaderOffsetScaleDefiner(ptCloud)
    export_lidar_points = las_header_offset_scale_definer.vectorized_get_coordinates_to_save_in_las(ptCloud)
    export_lidar_points.drop(columns=['x_lidar', 'y_lidar', 'z_lidar'], inplace=True)
    export_lidar_points = export_lidar_points.rename(columns={
        'new_x': 'X',
        'new_y': 'Y',
        'new_z': 'Z'
    }, copy=False)
    output_file = JakartoLasFile(output_filename, las_header_offset_scale_definer)
    # writing and closing las file
    output_file.write_and_close(export_lidar_points)


def exemple():
    data = {'gps_time': [0, 1.232, 2.543, 3.741],
            'intensity': [14578, 54236, 14265, 12543],
            'X': [456, 234, 567, 432],
            'Y': [10234, 10256, 10789, 10275],
            'Z': [10, 11, 12, 13]
            }

    dataframe = pd.DataFrame(data)
    pandas2las(dataframe, 'exemple.las')
