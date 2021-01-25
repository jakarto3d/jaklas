# jaklas

jaklas est un repo permettant d'enregistrer des tableaux pandas en nuage de point au format .las

(ou tout object qui implémente ``__getitem__``)

## Requirements
```bash
numpy
pandas
pylas
```

## Installation
```bash
git clone git@github.com:jakarto3d/jaklas.git
cd jaklas
python -m pip install .
```

### Tester l'installation
```bash
pip install -r requirements-dev.txt
pytest
```

## Utilisation
La fonction ``jaklas.write`` convertit un pandas dataframe en fichier las. 
Le dataframe **doit** avoir les champs:
 - x (ou X)
 - y (ou Y)
 - z (ou Z)
et il peut avoir les champs:
 - gps_time
 - intensity
 - classification
 - red
 - green
 - blue
 - tout autre champs personnalisé

Voir l'exemple :
```python
import jaklas
import pandas
data = {'gps_time': [0, 1.232, 2.543, 3.741],
        'intensity': [14578, 54236, 14265, 12543],
        'X': [456, 234, 567, 432],
        'Y': [10234, 10256, 10789, 10275],
        'Z': [10, 11, 12, 13]
        }
dataframe = pandas.DataFrame(data)
filename = 'exemple.las'
jaklas.write(dataframe, filename)
```

Voir les paramètres de la fonction ``jaklas.write`` pour plus de fonctionnalités:

 - gestion automatique du format de points dans le fichier las de sortie
 - data_min_max est utilisé pour mettre à l'échelle les données en fonction du format des champs
