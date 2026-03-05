from typing import Dict, Any, Union
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

from v2.data.tags import HDF5Tags, AttributeTags
from v2 import bool_env
from v2.data import generate_subject_id
from v2.configs import DataConfig
from v2.data import serialize_np


class Subject:
    """
    Represents a subject in the analysis pipeline and links to its HDF5 file.
    Contains methods for reading and writing HDF5 datasets.
    frame can be specified as an index or a name, if frame names are provided in the metadata.
    Since orientation in hdf5 visualization is flipped, all datasets are flipped upside down (along axis 1)
    before reading/writing.
    """

    def __init__(
        self,
        input_file: str,
        data_config: DataConfig,
        group_id: int,
        study_id: int,
        scan_id: int,
        frame: Union[int, str],
        metadata: Dict[str, Any] = None,
    ):
        self.input_file = input_file

        self.group_id = group_id
        self.study_id = study_id
        self.scan_id = scan_id
        self.frame = frame

        self._hdf5_path = None
        self._extraction_results = None

        self.src_metadata = {} if metadata is None else metadata
        self.data_config = data_config

        self.subject_id = generate_subject_id(group_id, study_id, scan_id, frame)

    def initialize_hdf5(self, input_dir: Path = None):
        """Create the HDF5 file and store raw input data and metadata."""

        # by default input dir is contained in data_config
        if input_dir is None:
            input_dir = Path(self.data_config.input_dir)

        self._hdf5_path = (
            Path(os.environ["RUN_OUTDIR"])
            / "data"
            / f"{Path(self.subject_id).stem}.hdf5"
        )

        if self._hdf5_path.exists() and not bool_env("OVERWRITE"):
            # If we skip initialization, we still want derived metadata (e.g. scan_date)
            # that was persisted into the existing HDF5 to be available on the Subject.
            with h5py.File(self._hdf5_path, "r") as f:
                persisted = json.loads(f.attrs.get(AttributeTags.METADATA, "{}"))
            self.src_metadata.update(persisted)
            logging.info(
                f"HDF5 for subject {self.subject_id} already exists. Skipping initialization (but loading metadata)."
            )
            return

        os.makedirs(self._hdf5_path.parent, exist_ok=True)

        try:
            # TODO: this could be moved into some import_dataset utility or Reader/Writer architecture (which could then also be used in DatasetLoader)
            if self.input_file.endswith(".nii"):
                raise NotImplementedError(
                    "Loading from NIfTI is not updated for the latest pipeline version."
                )
                # raw_oa_data, raw_us_data = self._load_raw_nifti_data(input_dir)
            elif self.input_file.endswith(".npy"):
                raise NotImplementedError(
                    "Loading from NPY is not updated for the latest pipeline version."
                )
                # raw_oa_data, raw_us_data = self._load_raw_npy_data(input_dir)
            elif self.input_file.endswith(".hdf5"):
                us_img, us_attr, us_meta, oa_img, oa_attr, oa_meta = (
                    self._load_hdf5_data(input_dir)
                )
            else:
                raise ValueError(
                    f"Unsupported file format for {self.input_file}. Only .nii, .npy and .hdf5 are supported."
                )
        except Exception as e:
            logging.error(
                f"Failed to load image for {self.input_file} (ID: {self.subject_id}): {e}"
            )
            return

        self.src_metadata[HDF5Tags.US] = us_attr
        self.src_metadata[HDF5Tags.OA] = oa_attr

        with h5py.File(self._hdf5_path, "w") as f:
            f.attrs[AttributeTags.INPUT_FILE] = str(self.input_file)
            f.attrs[AttributeTags.GROUP_ID] = self.group_id
            f.attrs[AttributeTags.STUDY_ID] = self.study_id
            f.attrs[AttributeTags.SCAN_ID] = self.scan_id
            f.attrs[AttributeTags.FRAME] = self.frame
            f.attrs[AttributeTags.METADATA] = json.dumps(
                serialize_np(self.src_metadata)
            )
            f.attrs[AttributeTags.DATA_CONFIG] = json.dumps(
                self.data_config.serialize()
            )

            raw_group = f.create_group(HDF5Tags.RAW)
            us_dataset = raw_group.create_dataset(
                HDF5Tags.US,
                data=us_img,
                compression="gzip",
                compression_opts=4,
            )

            # add meta
            for key, value in us_meta.items():
                us_dataset.attrs[key] = (
                    value if type(value) in [str, int, float] else str(value)
                )

            oa_dataset = raw_group.create_dataset(
                HDF5Tags.OA,
                data=oa_img,
                compression="gzip",
                compression_opts=4,
            )
            # add meta
            for key, value in oa_meta.items():
                oa_dataset.attrs[key] = (
                    value if type(value) in [str, int, float] else str(value)
                )

    def exists(self) -> bool:
        return self.hdf5_path.exists()

    def serialize(self) -> Dict:
        return {
            AttributeTags.SUBJECT_ID: self.subject_id,
            AttributeTags.INPUT_FILE: str(self.input_file),
            AttributeTags.HDF5_PATH: self._hdf5_path,
            AttributeTags.METADATA: self.src_metadata,
            AttributeTags.DATA_CONFIG: self.data_config.serialize(),
            AttributeTags.GROUP_ID: self.group_id,
            AttributeTags.STUDY_ID: self.study_id,
            AttributeTags.SCAN_ID: self.scan_id,
            AttributeTags.FRAME: self.frame,
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
            class_labels = self.src_metadata.get("class_labels", [])
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

    def _load_hdf5_data(self, input_dir: Path):
        # TODO add src meta to meta
        hdf5_path_oa = "/".join(self.data_config.hdf5_tags_oa)
        hdf5_path_us = "/".join(self.data_config.hdf5_tags_us)

        with h5py.File(input_dir / self.input_file, "r") as f:

            global_attr = dict(f.attrs)
            oa_data = f[hdf5_path_oa]
            oa_attr = dict(f[hdf5_path_oa].attrs)
            us_data = f[hdf5_path_us]
            us_attr = dict(f[hdf5_path_us].attrs)

            def _resolve_frame_ids(frame, avail_frames_us, avail_frames_oa):
                """
                THis can handle frame specified as index or name. If name is specified, it looks up the frame id in the hdf5 attributes frames list
                So we can also correctly index datasets where not all frames were reconstructed and match them to the us frame, assuming "frames" attribute is provided.
                """

                if isinstance(frame, (int, np.integer)):
                    # handle native ints and numpy integer types
                    return int(frame), int(frame)
                elif isinstance(frame, str):
                    frame_id_us = (
                        avail_frames_us.index(frame)
                        if frame in avail_frames_us
                        else None
                    )
                    frame_id_oa = (
                        avail_frames_oa.index(frame)
                        if frame in avail_frames_oa
                        else None
                    )

                    if frame_id_us is None or frame_id_oa is None:
                        raise ValueError(
                            f"Frame name '{frame}' not found in available frames for subject {self.subject_id}. OA frames: {avail_frames_oa}, US frames: {avail_frames_us}."
                        )

                    # if frame_id_oa != frame_id_us:
                    #     logging.info(
                    #         f"Frame ID for OA ({frame_id_oa}) and US ({frame_id_us}) do not match for subject {self.subject_id}"
                    #     )

                    return frame_id_us, frame_id_oa
                else:
                    raise ValueError(f"frame must be int or str. Got {frame!r}")

            def _resolve_target_channels(target_channels, avail_channels, default_len):
                # helper to get channel indices
                if target_channels is not None:
                    if all(isinstance(ch, (int, np.integer)) for ch in target_channels):
                        target_channel_ids = target_channels
                        target_channel_names = [
                            avail_channels[i] for i in target_channel_ids
                        ]
                    elif all(isinstance(ch, str) for ch in target_channels):
                        ids = [
                            avail_channels.index(ch) if ch in avail_channels else None
                            for ch in target_channels
                        ]
                        if None in ids:
                            raise ValueError(
                                f"Some channel names were not found in available channels for subject {self.subject_id}. Available channels: {avail_channels}, target channels: {target_channels}."
                            )
                        target_channel_ids = ids
                        target_channel_names = target_channels
                    else:
                        raise ValueError(
                            f"target_channels must be a list of either all ints or all strs. Got {target_channels}"
                        )
                else:
                    channel_ids = list(range(default_len))
                    target_channel_ids = channel_ids
                    target_channel_names = avail_channels

                return list(target_channel_ids), list(target_channel_names)

            # load the available frames and channels from the hdf5 attributes.
            # this should be provided in the hdf5 datasets for the raw data under frames and channel_names attributes,
            # however we also want to be able to handle legacy datasets where this was not provided
            # so if these attributes dont exist, they will be loaded from the metadata
            avail_frames_us = list(
                us_attr["frames"]
                if "frames" in us_attr
                else self.src_metadata["avail_frames_us"]
            )
            avail_frames_oa = list(
                oa_attr["frames"]
                if "frames" in oa_attr
                else self.src_metadata["avail_frames_oa"]
            )
            avail_channels_us = list(
                us_attr["channel_names"]
                if "channel_names" in us_attr
                else self.src_metadata["avail_channels_us"]
            )
            avail_channels_oa = list(
                oa_attr["channel_names"]
                if "channel_names" in oa_attr
                else self.src_metadata["avail_channels_oa"]
            )

            assert (
                len(avail_frames_us) == us_data.shape[0]
            ), f"Number of available US frames does not match number of frames in US data for subject {self.subject_id}"
            assert (
                len(avail_frames_oa) == oa_data.shape[0]
            ), f"Number of available OA frames does not match number of frames in OA data for subject {self.subject_id}"
            assert (
                len(avail_channels_us) == us_data.shape[1]
            ), f"Number of available US channels does not match number of channels in US data for subject {self.subject_id}"
            assert (
                len(avail_channels_oa) == oa_data.shape[1]
            ), f"Number of available OA channels does not match number of channels in OA data for subject {self.subject_id}"

            # infer frame ids
            frame_id_us, frame_id_oa = _resolve_frame_ids(
                self.frame, avail_frames_us, avail_frames_oa
            )

            # infer channel ids based on config and hdf5 attributes, default to all channels if not specified in config
            channel_ids_us, channel_names_us = _resolve_target_channels(
                self.data_config.target_channels_us,
                avail_channels_us,
                us_data.shape[1],
            )

            channel_ids_oa, channel_names_oa = _resolve_target_channels(
                self.data_config.target_channels_oa,
                avail_channels_oa,
                oa_data.shape[1],
            )

            assert len(channel_ids_us) == len(
                channel_names_us
            ), f"Number of target US channel ids does not match number of target US channel names for subject {self.subject_id}"
            assert len(channel_ids_oa) == len(
                channel_names_oa
            ), f"Number of target OA channel ids does not match number of target OA channel names for subject {self.subject_id}"

            # load the data into shape [channels, width_x, depth_z]
            us_img = us_data[frame_id_us, channel_ids_us, :, 0, :]
            oa_img = oa_data[frame_id_oa, channel_ids_oa, :, 0, :]

            us_meta = {
                "channel_names": channel_names_us,
                "channel_ids": channel_ids_us,
                "frame_id": frame_id_us,
                "frame_name": avail_frames_us[frame_id_us],
            }

            oa_meta = {
                "channel_names": channel_names_oa,
                "channel_ids": channel_ids_oa,
                "frame_id": frame_id_oa,
                "frame_name": avail_frames_oa[frame_id_oa],
            }

            # add global_attr to us_attr and oa_attr
            us_attr["global"] = global_attr
            oa_attr["global"] = global_attr

            return us_img, us_attr, us_meta, oa_img, oa_attr, oa_meta

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
