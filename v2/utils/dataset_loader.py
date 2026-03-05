import os
import numpy as np
from tqdm import tqdm
from v2.utils.ithera import load_iannotation

from v2.components.pipeline_component import PipelineComponent
from v2.data.tags import HDF5Tags


class DatasetLoader(PipelineComponent):
    """
    Loads an arbitrary dataset from a given path.
    Stores it under the given target group and dataset name.
    Since orientation in hdf5 visualization is flipped, all datasets are flipped upside down (along axis 1)
    """

    def _default_sources(self):
        return {}

    def _default_targets(self):
        return {
            "primary": (self.config.target_group, self.config.target_dataset),
        }

    def _run_component(self):

        # src_group, src_dataset = self.sources["primary"]
        target_group, target_dataset = self.targets["primary"]

        hdf5_attributes = self.config.serialize()
        hdf5_attributes["channel_names"] = self.config.channel_names

        pbar = tqdm(total=len(self.subjects), desc="Loading Datasets")

        for subject in self.subjects:
            pbar.set_description(
                f"Loading dataset file for subject: {subject.subject_id}"
            )

            filename = self.config.naming_scheme.format(
                group_id=subject.group_id,
                study_id=subject.study_id,
                scan_id=subject.scan_id,
                frame_id=subject.frame,
            )

            dataset_path = os.path.join(self.config.base_path, filename)

            if not os.path.exists(dataset_path):
                print(f"Dataset file {dataset_path} does not exist")
                continue

            # TODO: move this into a utility function (together with code in subject.py)
            if self.config.file_format == "nrrd":

                import nrrd

                data, _ = nrrd.read(dataset_path)
                data = data.squeeze()

                # must be 2d
                assert data.ndim == 2, "Data must be 2D for nrrd format"
                # add leading dimension
                data = data[np.newaxis, ...]
                # flip upside down to match hdf5 orientation
                data = np.flip(data, axis=1)

            elif self.config.file_format == "hdf5":

                import h5py

                with h5py.File(dataset_path, "r") as f:
                    data = f["data"][()]  # TODO: provide dataset name in config
                    data = data.squeeze()
                    # flip upside down to match hdf5 orientation
                    data = np.flip(data, axis=1)
            else:
                raise ValueError(f"Unsupported file format: {self.config.file_format}")

            subject.add_dataset(
                target_group,
                target_dataset,
                data,
                attributes=hdf5_attributes,
            )

            pbar.update(1)

        pbar.close()
