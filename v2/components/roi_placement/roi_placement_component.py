from tqdm import tqdm
import numpy as np

from v2.components.pipeline_component import PipelineComponent
from v2.components.roi_placement.roi_shapes import ShapeFactory
from v2.data.tags import HDF5Tags
from v2.utils.ithera import export_iannotation


class ROIPlacementComponent(PipelineComponent):
    """
    Places ROIs on segmentation masks based on specified shapes and configuration.
    """

    def _default_sources(self):
        return {
            "primary": (HDF5Tags.POSTPROCESSED, HDF5Tags.SEG),
        }

    def _default_targets(self):
        return {
            "primary": (HDF5Tags.ROI, HDF5Tags.SEG),
        }

    def _run_component(self):

        src_group, src_dataset = self.sources["primary"]
        target_group, target_dataset = self.targets["primary"]

        hdf5_attributes = self.config.serialize()
        hdf5_attributes["channel_names"] = [HDF5Tags.SEG]

        pbar = tqdm(total=len(self.subjects), desc="ROI Placement")

        for subject in self.subjects:
            pbar.set_description(
                f"ROI Placement Study {subject.study_id} Scan {subject.scan_id} Frame {subject.frame_id}"
            )

            mask, mask_meta = subject.load_dataset(src_group, src_dataset)

            # Validate and prepare mask
            if mask.ndim != 3 or mask.shape[0] != 1:
                raise ValueError(
                    f"Expected shape (1, H, W), got {mask.shape} for {subject.subject_id}"
                )

            # Remove the leading dim
            mask_2d = mask[0]

            # binarize mask to desired class
            mask_2d = (mask_2d == self.config.target_class_id).astype(np.uint8)

            # # overwrite global px_size by subject metadata
            # self.config.px_size = subject.metadata["px_size"]

            # Create ROI shape
            roi_shape = ShapeFactory.create_shape(self.config)

            # Get the ROI as mask and dict
            roi_mask, roi_dict = roi_shape.get_roi(mask_2d)

            if self.config.ilabs_export:
                if self.config.roi_type != "ellipse":
                    # TODO: implement other ROI types based on PATATO-Annotations
                    raise ValueError(
                        f"Exporting iannotation is currently only supported for ellipse ROIs, got {self.config.roi_type}"
                    )
                self._export_iannotation(
                    roi_dict,
                    self.output_dir
                    / f"{subject.study_id}_{subject.scan_id}.iannotation",
                )

            subject.add_dataset(
                target_group,
                target_dataset,
                roi_mask[np.newaxis, ...],
                attributes=hdf5_attributes,
            )

            pbar.update(1)

        pbar.close()

    @staticmethod
    def _export_iannotation(roi_dict, filepath):
        """Export ROI dict to ithera ilabs format."""
        export_iannotation(roi_dict, filepath)
