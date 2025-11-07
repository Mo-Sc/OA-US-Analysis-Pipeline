import os
from tqdm import tqdm
from datetime import datetime
import logging
import time

from v2.utils.setup import check_pipeline_config, save_config

from v2.pipeline import pipeline, global_config, data_config


def run_pipeline():

    # Check pipeline configuration
    check_pipeline_config(pipeline)

    # run_id is a timestamp, also used to create output folder.
    run_id = (
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if global_config.run_id is None
        else global_config.run_id
    )

    run_outpath = os.path.join(global_config.output_dir, run_id)
    os.makedirs(run_outpath, exist_ok=True)

    # Define logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(run_outpath, "pipeline.log")),
            logging.StreamHandler(),
        ],
        force=True,  # overrides previous basicConfig calls
    )

    # set some env vars that are used all over the project
    os.environ["RUN_ID"] = run_id
    os.environ["RUN_OUTDIR"] = run_outpath
    os.environ["OVERWRITE"] = str(global_config.overwrite).upper()
    os.environ["PX_SIZE"] = str(data_config.px_size)

    # save config to file
    config_file_path = os.path.join(run_outpath, "pipeline.json")
    save_config(global_config, data_config, pipeline, config_file_path)

    logging.info(f"##### STARTING SEG-CL-PIPELINE #####")
    logging.info(f"RUN ID: {run_id}")
    logging.info(f"OVERWRITE: {global_config.overwrite}")
    logging.info(f"OUTPUT BASEPATH: {run_outpath}\n")
    logging.info(f"STEPS: {[str(step.name) for step in pipeline]}")
    logging.info(f"Config file saved to {config_file_path}")
    logging.info(f"Found {len(pipeline[0].subjects)} subjects")

    logging.info(f"Loading subject data...")

    subjects_data = []

    for subject in tqdm(pipeline[0].subjects):
        try:
            subject.initialize_hdf5()
            subjects_data.append(subject)
        except Exception as e:
            logging.WARNING(f"Failed to initialize subject {subject.input_file}: {e}")

    logging.info(f"Created hdf5 files for {len(subjects_data)} subjects")

    logging.info(f"\nRunning pipeline...")

    pipeline_start_time = time.time()

    for step in pipeline:
        step_start_time = time.time()
        step.run()
        step_end_time = time.time()
        logging.info(
            f"{str(step.name)} completed in {step_end_time - step_start_time:.2f} seconds\n"
        )

    pipeline_end_time = time.time()
    logging.info(
        f"Pipeline completed in {pipeline_end_time - pipeline_start_time:.2f} seconds"
    )


if __name__ == "__main__":
    run_pipeline()
