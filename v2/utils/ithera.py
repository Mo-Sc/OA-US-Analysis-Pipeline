import math
import json


def px_roi_to_ithera(roi, px_size=0.0001, image_size=(400, 400)):
    """
    Used for exporting the roi to ithera format
    Takes ROI in pixel and cv2 ellipse coords and converts it to m and ithera coordinate system
    ithera coordinates: relative to the center of the image (top right is positive)

    """

    # Make sure values are floats
    roi_center = [float(value) for value in roi[0]]
    roi_axes_length = [float(value) for value in roi[1]]

    # shift ellipse center so it is relative to the center of the image
    roi_center = [roi_center[0] - image_size[0] / 2, roi_center[1] - image_size[1] / 2]

    # Calculate the position and size of the ROI
    roi_pos = [roi_center[0] - roi_axes_length[0], roi_center[1] - roi_axes_length[1]]
    roi_size = [roi_axes_length[0] * 2, roi_axes_length[1] * 2]

    # Convert the values to m
    roi_pos = [value * px_size for value in roi_pos]
    roi_size = [value * px_size for value in roi_size]

    area = math.pi * roi_size[0] / 2 * roi_size[1] / 2

    return roi_pos, roi_size, area


def ithera_roi_to_px(roi, px_size=0.0001, image_size=(400, 400), top_left=False):
    """
    Used for exporting the roi to ithera format
    Takes ROI in m and ithera coordinate system and converts it to pixel and cv2 ellipse coords
    roi: list of two lists / 2x3 np.array, first list contains the position of the ROI, second list contains the size of the ROI
    """

    # TODO: optional: shift back from center to top left corner

    # Make sure values are floats
    roi_pos = [float(value) for value in roi[0][0:2]]
    roi_size = [float(value) for value in roi[1][0:2]]

    # Calculate the center and half axes length of the ellipse
    roi_center = [roi_pos[0] + roi_size[0] / 2, roi_pos[1] + roi_size[1] / 2]
    roi_axes_length = [roi_size[0] / 2, roi_size[1] / 2]

    area = math.pi * roi_axes_length[0] * roi_axes_length[1]

    # Convert the values to pixels
    roi_center = [int(value / px_size) for value in roi_center]
    roi_axes_length = [int(value / px_size) for value in roi_axes_length]

    # shift ellipse center so it is relative to the top left corner of the image
    if top_left:
        roi_center = [
            roi_center[0] + image_size[0] / 2,
            roi_center[1] + image_size[1] / 2,
        ]

    return roi_center, roi_axes_length, area


def export_iannotation(roi_dict, outfile_path):
    """Export ROI information to ithera iAnnotation format."""

    rois = roi_dict["rois"]
    roi_types = roi_dict["roi_types"]
    annotated_frames = roi_dict["annotated_frames"]
    annotation_source = roi_dict["annotation_source"]

    if not isinstance(annotated_frames, list):
        annotated_frames = [12]

    roi_list_ian = []
    for roi, roi_type in zip(rois, roi_types):
        pos = roi[0]
        size = roi[1]
        roi_ian = {
            "__classname": "iROI",
            "type": roi_type,
            "pos": f"Point({pos[0]}, {pos[1]})",
            "size": f"Point({size[0]}, {size[1]})",
        }
        roi_list_ian.append(roi_ian)

    annotation = {
        "__classname": "iAnnotation",
        "ROIList": [roi_list_ian],
        "Sweeps": annotated_frames,
        "Source": annotation_source,
    }

    iannotation_dict = {
        "__classname": "iAnnotationListWrapper",
        "Annotations": [annotation],
        "ScanHash": "",
    }

    with open(f"{outfile_path}", "w") as f:
        json.dump(iannotation_dict, f, indent=4)


def load_iannotation(
    filepath, anno_index=-1, roi_index=-1, img_size=400, px_size=0.001
):
    """
    Load ROI data from ithera iannotation format
    """

    from dataset_tools.ithera.iannotation import IAnnotation

    try:

        iannotations = IAnnotation(filepath)
        # print(filepath)

        # extract the desired ROI
        roi = iannotations[anno_index]["ROIs"][roi_index]
        # create a binary mask from the ROI
        mask = roi.create_binary_mask(
            image_size=(img_size, img_size),
            px_size=px_size,
        )

        # mask is 2
        mask[mask == 1] = 2

    except:
        raise ValueError(f"Couldnt load ROI from iannotation file {filepath}")

    return mask
