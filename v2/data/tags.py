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

    # attrbiutes
    INPUT_FILE = "input_file"
    STUDY_ID = "study_id"
    SCAN_ID = "scan_id"
    FRAME_ID = "frame_id"
    GROUP_ID = "group_id"
    METADATA = "metadata"
    DATA_CONFIG = "data_config"
    GLOBAL_CONFIG = "global_config"
