import numpy as np


def generate_subject_id(group_id: int, study_id: int, scan_id: int, frame: int) -> str:
    """
    Generate a subject ID string based on the provided group, study, scan, and frame IDs.
    """

    return (
        f"{int(group_id):03d}-{int(study_id):03d}-{int(scan_id):03d}-{int(frame):03d}"
    )


def parse_subject_id(subject_id: str) -> tuple:
    """
    Parse a subject ID string into its components: group, study, scan, and frame IDs.
    """
    group_id, study_id, scan_id, frame_id = subject_id.split("-")
    return int(group_id), int(study_id), int(scan_id), int(frame_id)


def serialize_np(obj):
    """Recursively convert numpy arrays and scalars to Python types."""
    if isinstance(obj, dict):
        return {k: serialize_np(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_np(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(serialize_np(v) for v in obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # convert array to list
    elif isinstance(obj, np.generic):
        return obj.item()  # convert np.float32, np.int64, etc.
    else:
        return obj
