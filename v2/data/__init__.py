def generate_subject_id(
    group_id: int, study_id: int, scan_id: int, frame_id: int
) -> str:
    """
    Generate a subject ID string based on the provided group, study, scan, and frame IDs.
    """

    return f"{int(group_id):03d}-{int(study_id):03d}-{int(scan_id):03d}-{int(frame_id):03d}"


def parse_subject_id(subject_id: str) -> tuple:
    """
    Parse a subject ID string into its components: group, study, scan, and frame IDs.
    """
    group_id, study_id, scan_id, frame_id = subject_id.split("-")
    return int(group_id), int(study_id), int(scan_id), int(frame_id)
