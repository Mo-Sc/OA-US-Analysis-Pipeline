from typing import Dict, Any
from pathlib import Path
import h5py
import json
import ast
import os
import nibabel as nib
import datetime
import logging
import numpy as np
import imageio.v2 as imageio

from v2.data.tags import HDF5Tags
from v2 import bool_env
from v2.data import generate_subject_id


class Subject:
    """
    Represents a subject in the analysis pipeline and links to its HDF5 file.
    Contains methods for reading and writing HDF5 datasets.
    Since orientation in hdf5 visualization is flipped, all datasets are flipped upside down (along axis 1)
    before reading/writing.
    """

    def __init__(
        self,
        input_file: str,
        metadata: Dict[str, str],
        study_id: int,
        scan_id: int,
        frame_id: int,
        group_id: int,
    ):
        self.input_file = input_file
        self.metadata = metadata
        self.study_id = study_id
        self.scan_id = scan_id
        self.frame_id = frame_id
        self.group_id = group_id

        self._hdf5_path = None
        self._extraction_results = None

        self.subject_id = generate_subject_id(group_id, study_id, scan_id, frame_id)

    # def get_hdf5_path(self) -> Path:
    #     return os.path.join(os.environ["RUN_OUTDIR"] / f"{self.filename.stem}.hdf5")

    def initialize_hdf5(self, input_dir: Path = None):
        """Create the HDF5 file and store raw input data and metadata."""

        # by default input dir is contained in metadata
        if input_dir is None:
            input_dir = Path(self.metadata["input_dir"])

        self._hdf5_path = (
            Path(os.environ["RUN_OUTDIR"])
            / "data"
            / f"{Path(self.subject_id).stem}.hdf5"
        )

        if self._hdf5_path.exists() and not bool_env("OVERWRITE"):
            logging.info(
                f"HDF5 for subject {self.subject_id} already exists. Skipping initialization."
            )
            return

        os.makedirs(self._hdf5_path.parent, exist_ok=True)

        try:
            # TODO: this could be moved into some import_dataset utility (which could then also be used in DatasetLoader)
            if self.input_file.endswith(".nii"):
                raw_oa_data, raw_us_data = self._load_raw_nifti_data(input_dir)
            elif self.input_file.endswith(".npy"):
                raw_oa_data, raw_us_data = self._load_raw_npy_data(input_dir)
            elif self.input_file.endswith(".hdf5"):
                raw_oa_data, raw_us_data = self._load_raw_hdf5_data(input_dir)
            else:
                raise ValueError(
                    f"Unsupported file format for {self.input_file}. Only .nii, .npy and .hdf5 are supported."
                )
        except Exception as e:
            logging.error(
                f"Failed to load image for {self.input_file} (ID: {self.subject_id}): {e}"
            )
            return

        with h5py.File(self._hdf5_path, "w") as f:
            f.attrs[HDF5Tags.INPUT_FILE] = str(self.input_file)
            f.attrs[HDF5Tags.STUDY_ID] = self.study_id
            f.attrs[HDF5Tags.SCAN_ID] = self.scan_id
            f.attrs[HDF5Tags.FRAME_ID] = self.frame_id
            f.attrs[HDF5Tags.GROUP_ID] = self.group_id
            f.attrs[HDF5Tags.METADATA] = json.dumps(self.metadata)

            raw_group = f.create_group(HDF5Tags.RAW)
            oa_dataset = raw_group.create_dataset(
                HDF5Tags.OA,
                data=raw_oa_data,
                compression="gzip",
                compression_opts=4,
            )
            oa_dataset.attrs["channel_names"] = self.metadata["oa_channel_names"]

            us_dataset = raw_group.create_dataset(
                HDF5Tags.US,
                data=raw_us_data,
                compression="gzip",
                compression_opts=4,
            )
            us_dataset.attrs["channel_names"] = self.metadata["us_channel_names"]

    def exists(self) -> bool:
        return self.hdf5_path.exists()

    def serialize(self) -> Dict:
        return {
            "subject_id": self.subject_id,
            "filename": str(self.input_file),
            "hdf5_path": self._hdf5_path,
            "metadata": self.metadata,
            "study_id": self.study_id,
            "scan_id": self.scan_id,
            "frame_id": self.frame_id,
            "group_id": self.group_id,
        }

    def add_dataset(
        self,
        step_name: str,
        s_type: HDF5Tags,
        output_data: Any,
        attributes: Dict[str, Any] = None,
    ):
        """Save the output of each pipeline step into the HDF5 file and optionally add attributes."""

        assert (
            attributes["channel_names"] is not None
        ), "Channel names attribute is required for all datasets"

        with h5py.File(self.hdf5_path, "a") as f:
            if step_name in f:
                step_group = f[step_name]
            else:
                step_group = f.create_group(step_name)

            if s_type in step_group:
                if bool_env("OVERWRITE"):
                    del step_group[s_type]  # Delete the existing dataset to overwrite
                else:
                    logging.info(
                        f"Dataset '{s_type}' in step '{step_name}' already exists. Skipping."
                    )
                    return

            dataset = step_group.create_dataset(
                s_type,
                data=(
                    np.flip(output_data, 1)
                    if HDF5Tags.TABULAR not in s_type
                    else output_data
                ),  # flip for visualization except for tabular data
                compression="gzip",
                compression_opts=4,
            )
            step_group.attrs["timestamp"] = str(datetime.datetime.now())

            if attributes:
                for key, value in attributes.items():
                    dataset.attrs[key] = (
                        value if type(value) in [str, int, float] else str(value)
                    )

    def load_dataset(self, step_name: str, s_type: HDF5Tags) -> np.ndarray:
        """Load a dataset from the HDF5 file."""

        with h5py.File(self.hdf5_path, "r") as f:
            if step_name not in f:
                raise KeyError(f"Step '{step_name}' not found in HDF5.")

            dataset = f[step_name][s_type]

            dataset_np = (
                np.flip(np.array(dataset), 1)
                if HDF5Tags.TABULAR not in s_type
                else np.array(dataset)
            )  # flip back except for tabular data
            attributes = {key: dataset.attrs[key] for key in dataset.attrs}

            return dataset_np, attributes

    def export_dataset_to_png(
        self, step_name: str, s_type: HDF5Tags, output_dir: Path = None
    ):
        """Export central slice of dataset to PNG"""

        output_dir = os.environ["RUN_OUTDIR"] if not output_dir else output_dir
        os.makedirs(output_dir := output_dir / s_type, exist_ok=True)

        data, _ = self.load_dataset(step_name, s_type)

        # take central slice until 2D
        while data.ndim > 2:
            mid_slice = data.shape[0] // 2
            data = data[mid_slice, ...]

        # if dataset is segmentation, add legend
        if step_name == HDF5Tags.SEGMENTED:
            class_labels = self.metadata.get("class_labels", [])
            # TODO: add legend to image

        png_path = output_dir / f"{self.subject_id.stem}_{step_name}.png"
        imageio.imwrite(png_path, data)
        logging.info(f"Exported PNG for {step_name} to {png_path}")

    def export_dataset_to_nifti(
        self,
        step_name: str,
        s_type: HDF5Tags,
        output_dir: Path = None,
        affine: np.ndarray = None,
    ):
        """Export a dataset to a NIfTI file."""

        output_dir = os.environ["RUN_OUTDIR"] if not output_dir else output_dir
        os.makedirs(output_dir := output_dir / s_type, exist_ok=True)

        data, _ = self.load_dataset(step_name, s_type)

        new_img = nib.Nifti1Image(data, affine=affine, header=self.serialize())

        nifti_path = output_dir / f"{self.subject_id.stem}_{step_name}_{s_type}.nii.gz"
        nib.save(new_img, nifti_path)
        logging.info(f"Exported NIfTI for {step_name} - {s_type} to {nifti_path}")

    def _load_raw_nifti_data(self, input_dir: Path):

        raw_oa_data = nib.load(input_dir / "oa" / self.input_file).get_fdata()
        # move last dimension to the first position
        raw_oa_data = np.moveaxis(raw_oa_data, -1, 0)
        raw_us_data = nib.load(input_dir / "us" / self.input_file).get_fdata()
        # add leading dimension for US data
        raw_us_data = np.expand_dims(raw_us_data, axis=0)
        return np.flip(raw_oa_data, 1), np.flip(raw_us_data, 1)

    def _load_raw_npy_data(self, input_dir: Path):
        raw_oa_data = np.load(input_dir / "oa" / self.input_file)
        raw_us_data = np.load(input_dir / "us" / self.input_file)
        return np.flip(raw_oa_data, 1), np.flip(raw_us_data, 1)

    def _load_raw_hdf5_data(self, input_dir: Path):

        oa_hdf5_tags = "/".join(self.metadata["hdf5_tags"])

        with h5py.File(input_dir / self.input_file, "r") as f:
            oa_data = f[oa_hdf5_tags]
            oa_attr = f[oa_hdf5_tags].attrs
            us_data = f["ultrasounds/ultrasound/0"]

            # if hdf5 attributes has a timestamp ("date") field, add it to metadata
            if "date" in f.attrs:
                self.metadata["scan_date"] = f.attrs["date"]

            if self.metadata["hdf5_tags"][0] == "unmixed":
                # for unmixed data, use the only available frame in OA
                # and the given frame in the attributes for US
                # check if oa_attr has a "frame" attribute
                if "frame" in oa_attr:
                    # legacy, frame was not list yet
                    selected_frame = int(oa_attr["frame"])
                    oa_img = np.array(oa_data[self.frame_id, :, :, 0, :])
                    us_img = np.array(us_data[selected_frame, :1, :, 0, :])
                elif "frames" in oa_attr:
                    # new, frame is a list
                    # find position of target frame in list
                    avail_frames = ast.literal_eval(oa_attr["frames"])
                    selected_frame = avail_frames.index(self.frame_id)
                    oa_img = np.array(oa_data[selected_frame, :, :, 0, :])
                    us_img = np.array(us_data[self.frame_id, :1, :, 0, :])

            elif self.metadata["hdf5_tags"][0] == "reconstructions":
                # for reconstructions, use middle frame
                if "frame" in oa_attr:
                    # legacy, frame was not list yet
                    selected_frame = int(oa_attr["frame"])
                    oa_img = np.array(oa_data[self.frame_id, :, :, 0, :])
                    us_img = np.array(us_data[selected_frame, :1, :, 0, :])
                elif "frames" in oa_attr:
                    raise NotImplementedError(
                        "Frame selection for reconstructions with multiple frames not implemented yet."
                    )
                else:
                    # use middle frame
                    selected_frame = us_data.shape[0] // 2
                    oa_img = np.array(oa_data[selected_frame, :, :, 0, :])
                    us_img = np.array(us_data[selected_frame, :1, :, 0, :])

                # oa_img = np.array(
                #     oa_data[0, :, :, 0, :]
                # )  # todo remove and uncomment next

            return oa_img, us_img

    def extraction_results(
        self, src_group: str = HDF5Tags.EXTRACTED, src_dataset: str = HDF5Tags.TABULAR
    ):
        """
        Quick access to extracted feature intensities without IO
        """
        if self._extraction_results:
            return self._extraction_results
        else:
            try:
                extraction_results = self.load_dataset(src_group, src_dataset)
            except KeyError:
                raise ValueError(
                    f"Couldnt find extraction results for {self.subject_id}. Run an Extraction step first."
                )

            self._extraction_results = extraction_results
            return extraction_results

    @property
    def hdf5_path(self) -> Path:
        if self._hdf5_path is None:
            raise ValueError("HDF5 path not initialized. Call initialize_hdf5() first.")
        return self._hdf5_path

    @hdf5_path.setter
    def hdf5_path(self, path: Path):
        self._hdf5_path = path
