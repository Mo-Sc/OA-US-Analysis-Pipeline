import json
import os
import logging

from v2 import bool_env


def check_pipeline_config(pipeline):
    """Check if the pipeline configuration is valid."""
    if not pipeline:
        raise ValueError("Pipeline configuration is empty or invalid.")

    for step in pipeline:
        if not step.name or not step.config:
            raise ValueError(f"Step {step} is missing a name or configuration.")
        # todo: add more validation checks as needed
    logging.info("Pipeline configuration is valid.")


def save_config(global_config, data_config, pipeline, config_file_path):
    """Save the global configuration and pipeline configuration to a JSON file."""

    cfg = {}

    cfg["run_id"] = os.environ["RUN_ID"]
    cfg["run_outdir"] = os.environ["RUN_OUTDIR"]

    cfg["global_config"] = global_config.serialize()
    cfg["data_config"] = data_config.serialize()

    for step in pipeline:
        cfg[str(step.name)] = step.config.serialize()

        if bool_env("DEBUG"):
            cfg[str(step.name)]["subjects"] = [
                subject.serialize() for subject in step.subjects
            ]

    with open(config_file_path, "w") as config_file:
        json.dump(cfg, config_file, indent=4)
