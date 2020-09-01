import numpy as np


def sort(data, field: str, *args, **kwargs):
    try:
        # pandas.DataFrame
        return data.sort_values(by=field, *args, **kwargs)
    except AttributeError:
        pass

    argsort = np.argsort(data[field])
    return {k: v[argsort] for k, v in data.items()}
