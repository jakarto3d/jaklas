# jaklas

jaklas is a thin wrapper around [``pylas``](https://github.com/tmontaigu/pylas) to make reading and writing las files as
simple as possible.

The main use case is to write a pandas array to a las file in a single function call.
The las file attributes (point offset, point scaling, file version, point format, etc.)
are inferred depending on column names, datatype and point values.

The las writer supports any object implementing ``__getitem__`` that has the 
correct field names.

## Installation
```bash
pip install jaklas
```

### Testing
```bash
git clone git@github.com:jakarto3d/jaklas.git
cd jaklas
pip install -r requirements-dev.txt
python -m pip install .
pytest
```

## Usage
``jaklas.write`` writes a pandas dataframe (or a dict) to a las file.

The dataframe **must** have either (case insensitive):
- 'x', 'y' and 'z' columns 
- or an 'xyz' column

and it can have other las attributes (case sensitive names taken from ``pylas``):
 - gps_time
 - intensity
 - classification
 - red
 - green
 - blue
 - edge_of_flight_line
 - key_point
 - nir
 - number_of_returns
 - overlap
 - point_source_id
 - raw_classification
 - return_number
 - return_point_wave_location
 - scan_angle
 - scan_angle_rank
 - scan_direction_flag
 - scanner_channel
 - synthetic
 - user_data
 - wavepacket_index
 - wavepacket_offset
 - wavepacket_size
 - withheld
 - x_t
 - y_t
 - z_t

other column names will be written as extra dimensions.

## Example

```python
import jaklas
import pandas

data = {
    'gps_time': [0, 1.232, 2.543, 3.741],
    'intensity': [14578, 54236, 1425, 12543],
    'X': [456, 234, 567, 432],
    'Y': [10234, 10256, 10789, 10275],
    'Z': [10, 11, 12, 13],
}
dataframe = pandas.DataFrame(data)
filename = 'example.las'
jaklas.write(dataframe, filename)
```

Note the upper case 'X', 'Y' and 'Z' point data are the real coordinates,
not the scaled int32 ones like in the las file.

See [``jaklas.write``](https://github.com/jakarto3d/jaklas/blob/master/src/jaklas/write.py) docstring for more options like controlling offset and scaling.
