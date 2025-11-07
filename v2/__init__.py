import os
from enum import Enum, auto
import ast
import numpy as np


class StepNames(Enum):
    PREPROCESSING = auto()
    SEGMENTATION = auto()
    POSTPROCESSING = auto()
    ROI_PLACEMENT = auto()
    EXTRACTION = auto()
    ANALYSIS = auto()

    IANNOTATION_LOADING = auto()
    DATASET_LOADING = auto()
    SUMMARY_PLOT = auto()

    def __str__(self):
        return self.name


def bool_env(key: str, default: bool = False) -> bool:
    return os.environ.get(key, str(default)).lower() in ("1", "true", "yes", "on")


def attr_to_list(attr) -> list:
    """
    Convert a hdf5 attribute to a list.
    """
    if isinstance(attr, list):
        return attr
    elif isinstance(attr, np.ndarray):
        return attr.tolist()

    try:
        value = ast.literal_eval(attr)
        if isinstance(value, list):
            return value
        else:
            return [value]
    except (ValueError, SyntaxError):
        return [attr]
