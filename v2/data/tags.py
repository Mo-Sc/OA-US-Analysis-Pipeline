from v2 import StepNames


class HDF5Tags:

    # group names
    RAW = "RAW_DATA"
    PREPROCESSED = str(StepNames.PREPROCESSING)
    SEGMENTED = str(StepNames.SEGMENTATION)
    POSTPROCESSED = str(StepNames.POSTPROCESSING)
    ROI = str(StepNames.ROI_PLACEMENT)
    EXTRACTED = str(StepNames.EXTRACTION)
    IANNOTATION = "iannotations"

    # dataset or channel names
    OA = "OA"
    US = "US"
    SEG = "MASK"
    IANNOTATION = "IANNOTATION"
    TABULAR = "ROI_FEATURES"


class AttributeTags:
    INPUT_FILE = "input_file"
    HDF5_PATH = "hdf5_path"
    STUDY_ID = "study_id"
    SCAN_ID = "scan_id"
    FRAME = "frame"
    GROUP_ID = "group_id"
    SUBJECT_ID = "subject_id"
    METADATA = "source_metadata"
    DATA_CONFIG = "data_config"
    GLOBAL_CONFIG = "global_config"
