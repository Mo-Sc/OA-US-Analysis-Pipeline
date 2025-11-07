import numpy as np
from tqdm import tqdm
from skimage.morphology import remove_small_objects
from skimage.measure import label, regionprops
import logging
from typing import List

from v2.components.pipeline_component import PipelineComponent
from v2.data.tags import HDF5Tags


class PostprocessingComponent(PipelineComponent):
    """
    Performs postprocessing on segmentation masks.
    """

    def _default_sources(self):
        return {
            "primary": (HDF5Tags.SEGMENTED, HDF5Tags.SEG),
        }

    def _default_targets(self):
        return {
            "primary": (HDF5Tags.POSTPROCESSED, HDF5Tags.SEG),
        }

    def _run_component(self):

        src_group, src_dataset = self.sources["primary"]
        target_group, target_dataset = self.targets["primary"]

        pbar = tqdm(total=len(self.subjects), desc="Postprocessing")

        for subject in self.subjects:
            pbar.set_description(
                f"Postprocessing Study {subject.study_id} Scan {subject.scan_id} Frame {subject.frame_id}"
            )

            mask, mask_meta = subject.load_dataset(src_group, src_dataset)

            # Validate and prepare mask
            if mask.ndim != 3 or mask.shape[0] != 1:
                raise ValueError(
                    f"Expected shape (1, H, W), got {mask.shape} for {subject.subject_id}"
                )

            # Remove leading dim
            mask_2d = mask[0]

            processed_2d = self.process_mask(mask_2d)

            hdf5_attributes = self.config.serialize()
            hdf5_attributes["channel_names"] = mask_meta["channel_names"]

            subject.add_dataset(
                target_group,
                target_dataset,
                processed_2d[np.newaxis, ...],
                attributes=hdf5_attributes,
            )

            pbar.update(1)

        pbar.close()

    def process_mask(self, mask: np.ndarray) -> np.ndarray:
        cfg = self.config

        mask = self.keep_largest_region(mask, cfg.keep_largest_per_class)
        mask = self.reassign_freed_pixels_row_based(mask)
        mask = self.combine_classes(mask, cfg.combine_class_groups)

        for class_id, min_size in cfg.remove_small_objects_config:
            class_mask = mask == class_id
            processed_class = remove_small_objects(class_mask, min_size=min_size)
            mask[class_mask & ~processed_class] = 0

        mask = self.reassign_freed_pixels(mask)
        return mask

    @staticmethod
    def combine_classes(mask: np.ndarray, class_groups: List[List[int]]) -> np.ndarray:
        for group in class_groups:
            target = group[0]
            for cls in group[1:]:
                mask[mask == cls] = target
        return mask

    @staticmethod
    def keep_largest_region(mask: np.ndarray, class_ids: List[int]) -> np.ndarray:
        for class_id in class_ids:
            binary_mask = mask == class_id
            labeled_mask, num_labels = label(binary_mask, return_num=True)

            if num_labels > 1:
                regions = regionprops(labeled_mask)
                largest_region = max(regions, key=lambda r: r.area)
                largest_label = largest_region.label

                mask[(labeled_mask != largest_label) & binary_mask] = 0
        return mask

    @staticmethod
    def reassign_freed_pixels_row_based(mask: np.ndarray) -> np.ndarray:
        for i in range(mask.shape[0]):
            row = mask[i, :]
            freed_pixels = np.where(row == 0)[0]
            non_zero = row[row > 0]
            if len(non_zero) > 0:
                majority = np.bincount(non_zero).argmax()
                for j in freed_pixels:
                    mask[i, j] = majority
        return mask

    @staticmethod
    def reassign_freed_pixels(mask: np.ndarray) -> np.ndarray:
        freed_pixels = np.where(mask == 0)
        for i, j in zip(*freed_pixels):
            neighbors = mask[max(0, i - 1) : i + 2, max(0, j - 1) : j + 2].flatten()
            neighbors = neighbors[neighbors > 0]
            if len(neighbors) > 0:
                mask[i, j] = np.bincount(neighbors).argmax()
        return mask
