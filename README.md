# las_writer

las_writer est un repo permettant d'enregistrer des tableaux pandas en nuage de point au format .las

## Requirements
```bash
numpy
pandas
laspy
pathlib
```

## Installation
```bash
git clone git@github.com:jakarto3d/las_writer.git
cd las_writer
python -m pip install .
```
Tester l'installation
```python
import las_writer
las_writer.pandas2las.exemple()
```

## Utilisation
La fonction pandas2las convertit un pandas dataframe en fichier las. 
Le dataframe doit avoir les champs 'gps_time', 'intensity', 'X', 'Y', 'Z'.
Voir l'exemple :
```python
import las_writer
import pandas
data = {'gps_time': [0, 1.232, 2.543, 3.741],
        'intensity': [14578, 54236, 14265, 12543],
        'X': [456, 234, 567, 432],
        'Y': [10234, 10256, 10789, 10275],
        'Z': [10, 11, 12, 13]
        }
dataframe = pandas.DataFrame(data)
filename = 'exemple.las'
pandas2las(dataframe, filename)
```
