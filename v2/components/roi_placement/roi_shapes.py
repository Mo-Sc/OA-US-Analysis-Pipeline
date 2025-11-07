import os
from abc import abstractmethod, ABC
import logging

import numpy as np
import cv2

from v2.utils.ithera import px_roi_to_ithera


class ShapeFactory:
    """
    Factory class to create different ROI shapes.
    Supported shapes: Ellipse, Polygon, Static
    """

    @staticmethod
    def create_shape(config):
        if config.roi_type == "ellipse":
            return Ellipse(config.roi_params[config.roi_type])
        elif config.roi_type == "polygon":
            return Polygon(config.roi_params[config.roi_type])
        elif config.roi_type == "static":
            return Static(config.roi_params[config.roi_type])
        else:
            raise ValueError(f"Unknown shape type: {config.roi_type}")


class ROIShape(ABC):
    """
    Base class for ROI shapes.
    """

    px_size = float(os.environ.get("PX_SIZE", "0.0001"))

    def __init__(self, config):
        self.config = config

    @abstractmethod
    def get_roi(self, mask):
        """
        Places the ROI in the mask.
        Returns mask with ROI as np.array and ROI coordinates as dict.
        """
        pass


class Static(ROIShape):
    """
    Class representing a static ROI mask.
    The ROI is not computed dynamically — it is loaded from a file or directly provided in the configuration.
    """

    def __init__(self, config):
        super().__init__(config)

        # The static ROI must be provided either as a path or as a numpy array
        if "roi_mask" not in config:
            raise ValueError(
                "Static ROI requires 'roi_mask' in configuration (path to .npy/.png or numpy array)."
            )

        self.roi_mask = self._load_static_mask(config["roi_mask"])

    def _load_static_mask(self, mask_source):
        """
        Load the static ROI mask from a file path or directly from a numpy array.
        """
        if isinstance(mask_source, np.ndarray):
            mask = mask_source.astype(bool)
        elif isinstance(mask_source, str):
            if not os.path.exists(mask_source) or not mask_source.endswith(".npy"):
                raise ValueError(
                    f"Static ROI mask file {mask_source} does not exist or is not a .npy file."
                )
            mask = np.load(mask_source).astype(bool)

        return mask

    def get_roi(self, mask):
        """
        Apply the static ROI mask to the given segmentation mask.
        Returns the combined mask and an annotation dictionary.
        """
        assert (
            self.roi_mask.shape == mask.shape
        ), f"Static ROI mask shape {self.roi_mask.shape} does not match input mask shape {mask.shape}"

        if self.config["in_target_class"]:
            # place ROI where self.roi mask and mask are both True
            # this places the ROI (2) only where the segmentation mask (1) is present
            combined_mask = np.where((mask == 1) & (self.roi_mask == 1), 2, mask)
        else:
            # alternatively, return only static ROI (set to 2) and background (0)
            combined_mask = np.where(self.roi_mask == 1, 2, 0)

        # Generate a minimal annotation
        annotation_dict = {
            "annotated_frames": "default",
            "annotation_source": f"SEG-CL-Pipeline_{os.environ.get('RUN_ID', 'unknown')}",
            "rois": [],  # no shape-based ROI needed
            "roi_types": ["Static"],
        }

        return combined_mask, annotation_dict


class Ellipse(ROIShape):
    """
    Class to place an ellipse as ROI on the segmentation mask.

    """

    def get_roi(self, mask):
        """
        Get the ellipse ROI as mask and annotation dictionary.
        """

        # Place the ellipse in the mask
        mask_with_ellipse, center, axes = self._place_ellipse_in_mask(mask)

        roi = np.array([[center[0], center[1], 0.0], [axes[0], axes[1], 0.0]])

        # Convert the ROI to Ithera format for the iannotation file export
        ithera_roi_center, ithera_roi_axes, _ = px_roi_to_ithera(
            roi, px_size=self.px_size, image_size=mask.shape
        )

        # Create the annotation dictionary, structured similar to the Ithera format
        annotation_dict = {
            "annotated_frames": "default",
            "annotation_source": f"SEG-CL-Pipeline_{os.environ['RUN_ID']}",
            "rois": [
                [
                    [ithera_roi_center[0], ithera_roi_center[1], 0.0],
                    [ithera_roi_axes[0], ithera_roi_axes[1], 0.002],
                ]
            ],
            "roi_types": ["Ellipse"],
        }

        return mask_with_ellipse, annotation_dict

    def _place_ellipse_in_mask(self, mask):
        """
        Place an ellipse into the binary mask. The ellipse is centered on the horizontal axis
        and placed as far up as possible within the foreground mask, with an optional margin.
        mask: A binary numpy array where the foreground is labeled with 1s and the background with 0s.
        Returns copy of the original mask with the ellipse drawn on it.
        """

        # Ensure mask is binary
        mask = mask.astype(np.uint8)

        # Convert the ROI size from the config from m to pixels
        ellipse_width_px = int(self.config["roi_ellipse_size"][0] / self.px_size)
        ellipse_height_px = int(self.config["roi_ellipse_size"][1] / self.px_size)

        # Convert the margin from the config from m to pixels
        margin_px = int(self.config["margin"] / self.px_size)

        # Find the topmost point of the foreground band along the central vertical axis
        center_x = mask.shape[1] // 2
        for center_y in range(mask.shape[0]):
            if mask[center_y, center_x] == 1:
                break

        # Adjust the center_y by the given margin
        center_y += margin_px

        # Calculate ellipse parameters
        center = (
            center_x,
            center_y + ellipse_height_px // 2,
        )  # Centered on horizontal axis, placed as far up as possible with margin
        axes = (
            ellipse_width_px // 2,
            ellipse_height_px // 2,
        )  # Use the provided width and height

        # Create a copy of the mask to draw the ellipse on
        mask_with_ellipse = mask.copy()

        # Draw the ellipse on the mask
        cv2.ellipse(
            mask_with_ellipse,
            center,
            axes,
            angle=0,
            startAngle=0,
            endAngle=360,
            color=2,  # Use a distinct color value for the ellipse
            thickness=-1,
        )
        return mask_with_ellipse, center, axes


class Polygon(ROIShape):
    """
    Class to place a polygon as ROI on the segmentation mask.
    """

    def get_roi(self, mask):
        """
        Get the polygon ROI as mask and annotation dictionary.
        """

        trimmed_mask = mask.copy()

        # Trim the ROI to the desired height. This should keep the size of the ROI approximately constant.
        trimmed_mask = self._trim_roi(trimmed_mask)

        # overlay seg mask (1) and generated ROI (2)
        seg_mask_roi_overlay = np.where(trimmed_mask, 2, mask)

        return seg_mask_roi_overlay, {}

    def _trim_roi(self, mask):
        """
        Trims the ROI object in the segmentation mask to a given height and width.
        mask: Segmentation mask containing ROI (True) and background (False).
        Returns Modified segmentation mask with the ROI trimmed.
        """

        # Convert the desired height and width from meters to pixels
        height_px = int(self.config["roi_height"] / self.px_size)
        width_px = int(self.config["roi_width"] / self.px_size)

        # Find the vertical center of the image (center column)
        center_column = mask.shape[1] // 2

        # Determine the starting row of the foreground object in the center column
        foreground_rows = np.where(mask[:, center_column])[0]
        if len(foreground_rows) == 0:
            return mask  # No foreground found, return the original mask

        start_row = foreground_rows[0]
        end_row = start_row + height_px

        # Calculate the horizontal bounds for the ROI
        half_width = width_px // 2
        left_column = max(
            0, center_column - half_width
        )  # Ensure bounds don't exceed image
        right_column = min(
            mask.shape[1], center_column + half_width + 1
        )  # +1 for inclusive range

        # Trim the foreground object
        trimmed_mask = mask.copy()
        trimmed_mask[end_row:, :] = False  # Trim vertically
        trimmed_mask[:, :left_column] = False  # Trim horizontally (left side)
        trimmed_mask[:, right_column:] = False  # Trim horizontally (right side)

        return trimmed_mask
