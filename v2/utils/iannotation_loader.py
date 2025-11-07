import os
import numpy as np
from tqdm import tqdm
from v2.utils.ithera import load_iannotation

from v2.components.pipeline_component import PipelineComponent
from v2.data.tags import HDF5Tags


class IAnnotationLoader(PipelineComponent):
    """
    Loads ROI from ithera iannotation files and stores them in the pipeline hdf5.
    """

    def _default_sources(self):
        return {}

    def _default_targets(self):
        return {
            "primary": (HDF5Tags.ROI, HDF5Tags.IANNOTATION),
        }

    def _run_component(self):

        # src_group, src_dataset = self.sources["primary"]
        target_group, target_dataset = self.targets["primary"]

        hdf5_attributes = self.config.serialize()
        hdf5_attributes["channel_names"] = [HDF5Tags.IANNOTATION]

        pbar = tqdm(total=len(self.subjects), desc="ROI Placement")

        for subject in self.subjects:
            pbar.set_description(
                f"Loading iannotation file for Study {subject.study_id} Scan {subject.scan_id} Frame {subject.frame_id}"
            )

            # Load mask from ithera iannotation (stored in hdf5)
            iannotation_path = os.path.join(
                self.config.base_path,
                f"Study_{subject.study_id}/Scan_{subject.scan_id}.hdf5",
                # f"Study_{subject.study_id}/Scan_{subject.scan_id}.iannotation",
            )
            mask_2d = self._load_iannotation(iannotation_path)

            subject.add_dataset(
                target_group,
                target_dataset,
                mask_2d[np.newaxis, ...],
                attributes=hdf5_attributes,
            )

            pbar.update(1)

        pbar.close()

    def _load_iannotation(self, filepath):
        """Load ROI dict from ithera ilabs format."""
        return load_iannotation(
            filepath,
            self.config.anno_index,
            self.config.roi_index,
            self.config.img_size,
            self.config.px_size,  # replace by environ
        )
