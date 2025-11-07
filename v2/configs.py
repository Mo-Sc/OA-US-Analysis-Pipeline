from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from pathlib import Path
from abc import ABC
import numpy as np
from v2.data.tags import HDF5Tags


class Config(ABC):
    """
    Base class for all configuration classes.
    """

    def serialize(self):
        """
        recursively serializes everything that inherits from Config
        Useful for json or hdf5 attributes
        Arrays that are part of the config are omitted to avoid large attributes.
        """

        def _serialize_value(value):
            if isinstance(value, Config):
                return value.serialize()
            elif isinstance(value, Path):
                return str(value)
            elif isinstance(value, (list, tuple)):
                return [_serialize_value(v) for v in value]
            elif isinstance(value, dict):
                return {k: _serialize_value(v) for k, v in value.items()}
            elif isinstance(value, np.ndarray):
                return f"ndarray shape {value.shape}, dtype {value.dtype}"
            else:
                return value

        return {key: _serialize_value(value) for key, value in self.__dict__.items()}


# --- Configuration Classes for the Pipeline --
# Contains default values as used for the SENDERO PAD dataset


@dataclass
class GlobalConfig(Config):
    output_dir: Path = Path("data/output")
    logging_level: str = "INFO"
    overwrite: bool = True
    random_seed: int = 42
    run_id: Optional[str] = None


@dataclass
class DataConfig(Config):
    """
    Configuration for data loading
    Required parameters depend on the dataset
    """

    input_dir: Path = Path("data/input")
    metadata_file: Path = Path("data/metadata.csv")
    hdf5_tags: Optional[List[str]] = field(default_factory=list)
    target_scans: Optional[List[int]] = None  # None means all scans
    target_frames: Optional[List[int]] = None  # None means all frames
    us_channel_names: List[str] = field(default_factory=list)
    oa_channel_names: List[str] = field(default_factory=list)
    px_size: float = 0.0001  # pixel size in m, image is 4cm x 4cm


@dataclass
class PreprocessingConfig(Config):
    """
    Configuration for preprocessing images.
    Contains default values for the preprocessing step.

    Parameters:
    - flipud: bool: flip img upside down.
    - fliplr: bool: flip img left to right.
    - rotate90: int: rotate img n times 90 degrees.
    - img_size: int: resize img to img_size x img_size.
    """

    flipud: bool = False
    fliplr: bool = False
    rotate90: int = 0
    img_size: int = 400


@dataclass
class SegmentationConfig(Config):
    """
    Configuration for segmentation models.

    Parameters:
    - `arch`: Architecture of the segmentation model (e.g., "cUNet", "nnUNet").
    - `model_config`: Configuration for the specific model architecture.
    """

    arch: str = "cUNet"
    model_config: Optional[Config] = None


@dataclass
class PostprocessingConfig(Config):
    """
    Configuration for postprocessing segmentation masks.

    In the following order:
    - `keep_largest_per_class`: List of class IDs for which to keep only the largest connected component.
    - `combine_class_groups`: List of class ID groups to be merged. The first ID in each group will be the target class.
    - `remove_small_objects_config`: List of (class_id, min_size) tuples for removing small regions per class.
    """

    # default for Haut, Muskel1, Vorlaufstrecke and SAT
    keep_largest_per_class: List[int] = field(default_factory=lambda: [1, 3, 7, 8])
    combine_class_groups: List[List[int]] = field(
        default_factory=lambda: [
            [2, 6],  # Faszie1 + Faszie2 → 2
            [3, 4],  # Muskel1 + Muskel2 → 3
        ]
    )
    # default for Haut, Muskel1, Membran, Vorlaufstrecke, SAT and Gel (min size 50)
    # and for Faszie1 (min size 25)
    remove_small_objects_config: List[Tuple[int, int]] = field(
        default_factory=lambda: [
            (1, 50),
            (3, 50),
            (5, 50),
            (7, 50),
            (8, 50),
            (9, 50),
            (2, 25),
        ]
    )


@dataclass
class ROIPlacementConfig(Config):
    """
    Configuration for ROI placement.

    Parameters:
    - `target_class_id`: in which class to place the ROI.
    - `roi_type`: Type of ROI, can be "ellipse" or "polygon" or "static".
    - `roi_params`: Dictionary containing parameters specific to the ROI type.
    - `iannotation_export`: Whether to export ROIs in iLabs format.
    """

    target_class_id: int = 3
    roi_type: str = "ellipse"
    roi_params: dict = field(
        default_factory=lambda: {
            "ellipse": {
                "roi_ellipse_size": [0.00960219, 0.00244162],  # in meters
                "margin": 0.0,  # Depth of ellipse in muscle (in meters)
            },
            "polygon": {
                "roi_height": 0.00244162,  # in meters
                "roi_width": 0.015,  # in meters
            },
            "static": {
                "roi_mask": "roi_mask.npy",  # path to a binary mask .npy file or a numpy array
                "in_target_class": False,  # if True, place ROI only within target class
            },
        }
    )
    ilabs_export: bool = False  # Export ROIs in iLabs format


@dataclass
class ExtractionConfig(Config):
    """
    Configuration for intensity extraction.
    Parameters:
    - `target_class_id`: Class ID of the target ROI for extraction.
     # see https://pyradiomics.readthedocs.io/en/latest/features.html for available features
     # Export a combined xlsx file with extracted features for all subjects
    derived channels supports mSO2 or a ratio of two channels, e.g. "700/850" for 700nm/850nm ratio.
    """

    # TODO: here a channel selection would make sense already
    target_class_id: int = 2  # default target class id from roi_placement component
    feature_classes: List[str] = field(default_factory=lambda: ["firstorder"])
    derived_channels: Optional[List[str]] = field(default_factory=list)
    xlsx_export: bool = True


@dataclass
class AnalysisConfig(Config):
    """
    Configuration for analysis strategies.

    Parameters:
    - `strategy_name`: The analysis that should be performed (see components/analysis/strategy_configs.py).
    - `strategy_config`: Configuration for the specific strategy.
    """

    strategy_name: str = "group_comparison"
    strategy_config: Optional[Config] = None


@dataclass
class IAnnotationLoaderConfig(Config):
    """
    Configuration for loading ithera specific iAnnotation files.
    The selected annotation ROI is converted into a mask and stored as ROI_PLACEMENT dataset by default
    """

    base_path: Path = Path("data/input")
    anno_index: int = -1
    roi_index: int = -1
    img_size: int = 400
    px_size: float = 0.0001


@dataclass
class DatasetLoaderConfig(Config):
    """
    Loads an arbitrary dataset from a given path.
    Stores it under the given target group and dataset name.
    """

    base_path: Path = Path("data/input")
    target_group: str = "RAW"
    target_dataset: str = "DATASET"
    channel_names: List[str] = field(default_factory=lambda: ["700", "850", "980"])
    naming_scheme: str = "{study_id}_{scan_id}.nrrd"
    file_format: str = "nrrd"  # or "nifti", "npy", etc.


@dataclass
class SummaryPlotConfig(Config):
    """
    Creates a PNG containing plots from all pipeline steps in plottable plotable_groups.
    """

    plotable_groups: List[str] = field(
        default_factory=lambda: [
            HDF5Tags.RAW,
            # HDF5Tags.PREPROCESSED,
            HDF5Tags.SEGMENTED,
            HDF5Tags.POSTPROCESSED,
            HDF5Tags.ROI,
            HDF5Tags.EXTRACTED,
        ]
    )
    oa_channel_name: str = "Hb"  # which oa channel to plot (only one for performance)
    plot_size: int = 400  # size of a single plot in pixels
