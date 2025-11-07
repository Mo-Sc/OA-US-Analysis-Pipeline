import os
import re
import pandas as pd
from .subject import Subject
from . import parse_subject_id

# contains scripts to import different datasets into the pipeline structure


def collect_subjects_demo(data_config, file_ending=".npy"):
    """
    Creates subjects based on files in the basepath with the given file ending.
    Suitable for legacy datasets
    Adds study_id, scan_id. Group and frame are constant for all subjects.
    """

    subjects = []
    for i, filename in enumerate(os.listdir(os.path.join(data_config.input_dir, "us"))):
        if filename.endswith(file_ending):

            assert os.path.exists(
                os.path.join(data_config.input_dir, "oa", filename)
            ), f"US file {filename} does nothave corresponding OA file"

            study_id = filename.split("_")[0]
            scan_id = filename.split("_")[1].split(".")[0]
            group_id = i % 2  # alternate between group 0 and 1 (demo)
            frame_id = 0

            subjects.append(
                Subject(
                    input_file=filename,
                    metadata=data_config.serialize(),
                    study_id=study_id,
                    scan_id=scan_id,
                    frame_id=frame_id,
                    group_id=group_id,
                )
            )

    return subjects


def resume_subjects(data_config):
    """
    Loads subjects from a previous run
    """

    subjects = []
    for filename in os.listdir(data_config.input_dir):
        if filename.endswith(".hdf5"):
            # parse the subject id
            subject_id = filename.split(".")[0]
            group_id, study_id, scan_id, frame_id = parse_subject_id(subject_id)

            subjects.append(
                Subject(
                    input_file=filename,
                    metadata=data_config.serialize(),
                    study_id=study_id,
                    scan_id=scan_id,
                    frame_id=frame_id,
                    group_id=group_id,
                )
            )
    return subjects


def get_avail_frame_ids(hdf5_path, hdf5_tags):
    """
    Returns a list of available frame ids in the hdf5 file
    """

    import h5py
    import ast

    with h5py.File(hdf5_path, "r") as f:
        # navigate to the dataset using the tags
        dataset = f["/".join(hdf5_tags)]
        # check whether dataset has a frames attribute
        if "frames" in dataset.attrs:
            # unmixed datasets usually have a frames attribute, indicating which of the original frames were used for unmixing
            frame_ids = ast.literal_eval(dataset.attrs["frames"])
            if dataset.shape[0] < len(frame_ids):
                print(
                    f"WARNING: target frames list has more entries than frames in the dataset for {hdf5_path}"
                )
        else:
            # otherwise use all available frames
            frame_ids = list(range(dataset.shape[0]))

    return frame_ids
