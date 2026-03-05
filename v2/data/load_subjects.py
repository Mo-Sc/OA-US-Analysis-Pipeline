import os
import re
import pandas as pd
from .subject import Subject
from . import parse_subject_id
import h5py
import numpy as np
import logging

# contains scripts to import different datasets into the pipeline structure

def collect_subjects_demo(data_config):
    """
    Creates subjects based on files in the basepath with the given file ending.
    """

    subjects = []
    for i, filename in enumerate(os.listdir(data_config.input_dir)):
        if filename.endswith(".hdf5"):

            group_id, study_id, scan_id, frame = parse_subject_id(
                filename.split(".")[0]
            )

            subjects.append(
                Subject(
                    input_file=filename,
                    data_config=data_config,
                    group_id=group_id,
                    study_id=study_id,
                    scan_id=scan_id,
                    frame=frame,
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
                    data_config=data_config,
                    group_id=group_id,
                    study_id=study_id,
                    scan_id=scan_id,
                    frame=frame_id,
                )
            )
    return subjects

