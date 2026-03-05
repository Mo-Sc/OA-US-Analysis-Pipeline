import os
import logging

import numpy as np
from tqdm import tqdm
from skimage.transform import resize

from ..pipeline_component import PipelineComponent
from ...data.tags import HDF5Tags


class PreprocessingComponent(PipelineComponent):
    """
    Preprocesses the input data.
    """

    def _default_sources(self):
        return {
            "primary": (HDF5Tags.RAW, HDF5Tags.US),
        }

    def _default_targets(self):
        return {
            "primary": (HDF5Tags.PREPROCESSED, HDF5Tags.US),
        }

    def _run_component(self):

        src_group, src_dataset = self.sources["primary"]
        target_group, target_dataset = self.targets["primary"]

        pbar = tqdm(total=len(self.subjects), desc="Preprocessing")

        for subject in self.subjects:

            pbar.set_description(f"Preprocessing Study subject: {subject.subject_id}")

            # Load the scan data
            data, data_meta = subject.load_dataset(src_group, src_dataset)

            # flip / mirror / rotate the scan data if necessary
            if self.config.flipud:
                data = np.flip(data, axis=1)

            if self.config.fliplr:
                data = np.flip(data, axis=2)

            if self.config.rotate90:
                data = np.rot90(data, self.config.rotate90, axes=(1, 2))

            # resize image if necessary (always assumed to be square)
            if self.config.img_size != data.shape[-1]:
                if self.hdf5_attr_list(data_meta["channel_names"]) == [HDF5Tags.SEG]:
                    # use nearest neighbor interpolation for masks
                    data = resize(
                        data,
                        (data.shape[0], self.config.img_size, self.config.img_size),
                        order=0,
                        mode="constant",
                        preserve_range=True,
                        anti_aliasing=False,
                    )
                else:
                    data = resize(
                        data,
                        (data.shape[0], self.config.img_size, self.config.img_size),
                        order=3,
                        mode="constant",
                        preserve_range=True,
                        anti_aliasing=True,
                    )

            hdf5_attributes = self.config.serialize()
            hdf5_attributes["channel_names"] = data_meta["channel_names"]

            # save the preprocessed data
            subject.add_dataset(
                target_group,
                target_dataset,
                data,
                attributes=hdf5_attributes,
            )

            pbar.update(1)
        pbar.close()
