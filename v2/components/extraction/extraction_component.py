import numpy as np
from tqdm import tqdm
import logging
import os
from collections import defaultdict
from radiomics.featureextractor import RadiomicsFeatureExtractor
import SimpleITK as sitk

from v2.components.pipeline_component import PipelineComponent
from v2.data.tags import HDF5Tags
from v2 import bool_env
from v2.utils.channels import calculate_mso2, calculate_ratio


class ExtractionComponent(PipelineComponent):
    """
    Performs intensity extraction from OA scan in given ROI.
    """

    px_size = float(os.environ.get("PX_SIZE", "0.0001"))

    def _default_sources(self):
        return {
            "roi": (HDF5Tags.ROI, HDF5Tags.SEG),
            "oa": (HDF5Tags.RAW, HDF5Tags.OA),
        }

    def _default_targets(self):
        return {
            "features": (HDF5Tags.EXTRACTED, HDF5Tags.TABULAR),
            "oa": (HDF5Tags.EXTRACTED, HDF5Tags.OA),
        }

    def _run_component(self):

        src_group_roi, src_dataset_roi = self.sources["roi"]
        src_group_oa, src_dataset_oa = self.sources["oa"]
        target_group_features, target_dataset_features = self.targets["features"]
        target_group_oa, target_dataset_oa = self.targets["oa"]

        # initialize the radiomics feature extractor
        extractor = RadiomicsFeatureExtractor(additionalInfo=False)

        # oppress pyradiomics logging
        logging.getLogger("radiomics").setLevel(logging.WARNING)

        # enable only the specified feature classes
        if self.config.feature_classes:
            extractor.disableAllFeatures()
            for feature_class in self.config.feature_classes:
                extractor.enableFeatureClassByName(feature_class)

        self.feature_extractor = extractor

        if self.config.xlsx_export:
            combined_intensity_dict = defaultdict(dict)

        pbar = tqdm(total=len(self.subjects), desc="Postprocessing")

        for subject in self.subjects:
            pbar.set_description(
                f"Intensity Extraction, Study {subject.study_id} Scan {subject.scan_id} Frame {subject.frame_id}"
            )

            roi, _ = subject.load_dataset(src_group_roi, src_dataset_roi)
            oa, oa_meta = subject.load_dataset(src_group_oa, src_dataset_oa)

            try:
                # if channel names are stored in hdf5 attributes convert to list
                oa_channels = self.hdf5_attr_list(oa_meta["channel_names"])
            except KeyError:
                # this uses hardcoded fallback for known channel configs
                # necessary for old datasets without channel names in metadata
                # TODO: remove this fallback in future versions
                logging.warning(
                    f"Channel names not found in metadata for {subject.subject_id}. Falling back to hardcoded channel names."
                )
                from v2.utils.channels import infer_channel_names_from_shape

                oa_channels = infer_channel_names_from_shape(oa.shape)

                logging.warning(f"Channel names: {oa_channels}")

            if roi.shape[1:] != oa.shape[1:]:
                raise ValueError(
                    f"Shape of ROI mask does not match OA scan: {roi.shape} != {oa.shape}"
                )

            if oa.shape[0] != len(oa_channels):
                raise ValueError(
                    f"Shape of OA scan doenst match the subjects channel names: {oa.shape[0]} != {len(oa_channels)}"
                )

            for derived_channel in self.config.derived_channels:
                if "/" in derived_channel:
                    # calculate ratio of two channels
                    ch1, ch2 = derived_channel.split("/")
                    if ch1 not in oa_channels or ch2 not in oa_channels:
                        raise ValueError(
                            f"Channels {ch1} and {ch2} not found in OA channels: {oa_channels}"
                        )
                    ratio = calculate_ratio(
                        oa[oa_channels.index(ch1)],
                        oa[oa_channels.index(ch2)],
                        clip=[-10, 10],
                        zero_nan=True,
                    )
                    oa_channels.append(derived_channel)
                    oa = np.concatenate((oa, np.expand_dims(ratio, axis=0)), axis=0)
                elif derived_channel == "mSO2":
                    # mSO2 is calculated from Hb and HbO2 channels
                    if "Hb" not in oa_channels or "HbO2" not in oa_channels:
                        raise ValueError(
                            "mSO2 extraction requires Hb and HbO2 channels to be present in OA scan."
                        )
                    # calculate mSO2 from Hb and HbO2
                    mSO2 = calculate_mso2(
                        oa[oa_channels.index("Hb")],
                        oa[oa_channels.index("HbO2")],
                        clip=True,
                    )
                    oa_channels.append("mSO2")
                    # add mSO2 channel to the OA scan
                    oa = np.concatenate((oa, np.expand_dims(mSO2, axis=0)), axis=0)

            channel_intensities = []
            roi_oa = []  # masked oa data
            for i, oa_channel in enumerate(oa_channels):

                roi_features = self._summarize_roi_intensity(roi[0], oa[i])

                # extract mask content for the target class id

                oa_masked = np.where(roi[0] == self.config.target_class_id, oa[i], 0)
                roi_oa.append(oa_masked)

                channel_intensities.append(roi_features)

                if self.config.xlsx_export:
                    combined_intensity_dict[oa_channel][
                        subject.subject_id
                    ] = roi_features

            # stack channel intensities into np array
            roi_features_np = np.stack(
                [
                    np.array(list(roi_features.values()))
                    for roi_features in channel_intensities
                ],
                axis=0,
            )

            hdf5_attributes = self.config.serialize()
            hdf5_attributes["target_class_id"] = self.config.target_class_id
            hdf5_attributes["channel_names"] = oa_channels

            roi_oa_np = np.stack(roi_oa, axis=0)

            # add the masked OA data to the subject
            subject.add_dataset(
                target_group_oa,
                target_dataset_oa,
                roi_oa_np,
                attributes=hdf5_attributes,
            )

            features_names = list(channel_intensities[0].keys())
            hdf5_attributes["features"] = features_names

            subject.add_dataset(
                target_group_features,
                target_dataset_features,
                roi_features_np,
                attributes=hdf5_attributes,
            )

            pbar.update(1)

        pbar.close()

        if self.config.xlsx_export:
            self._export_xlsx(combined_intensity_dict)
            logging.info(f"intensity extraction xlsx exported to {self.output_dir}")

    def _summarize_roi_intensity(self, mask, scan):
        """
        Overlaps the ROI mask with the OA scan and summarizes intensity values in ROI.
        """

        roi_mask = mask.copy()
        oa_scan = scan.copy()

        # extract only the ROI
        # roi_mask[roi_mask != self.config.target_class_id] = 0

        # create sitk images
        sitk_mask = sitk.GetImageFromArray(np.expand_dims(roi_mask, axis=0))
        sitk_scan = sitk.GetImageFromArray(np.expand_dims(oa_scan, axis=0))

        # use radiomics to extract features
        roi_features = self.feature_extractor.execute(
            sitk_scan, sitk_mask, label=self.config.target_class_id
        )

        # calculates the centroid by averaging the coordinates of all pixels in the ROI
        roi_centroid_px = np.mean(
            np.argwhere(roi_mask == self.config.target_class_id), axis=0
        )
        # convert to m
        roi_centroid = roi_centroid_px * self.px_size
        # row index (-> z coord) is first
        roi_features["roi-centroid-z"] = roi_centroid[0]
        roi_features["roi-centroid-x"] = roi_centroid[1]
        roi_features["roi-centroid-y"] = 0.0  # for now only 2D

        return roi_features

    def _export_xlsx(self, combined_intensity_dict):
        """
        Export the combined intensity dict to an XLSX file, including a 'mean' row for each channel.
        """

        import pandas as pd

        filename = (
            f"intensity_extraction_"
            f"{self.sources['oa'][0].replace('/', '_')}_{self.sources['oa'][1]}_"
            f"{self.sources['roi'][0].replace('/', '_')}_{self.sources['roi'][1]}.xlsx"
        )
        outfile = self.output_dir / filename

        if os.path.exists(outfile) and not bool_env("OVERWRITE"):
            logging.info(f"XLSX file {outfile} already exists. Skipping export.")
            return

        dfs_by_channel = {}

        for channel, subjects in combined_intensity_dict.items():
            # Convert dict of dicts to DataFrame
            df = pd.DataFrame.from_dict(subjects, orient="index").reset_index()
            df = df.rename(columns={"index": "subject_id"})

            # Convert any 0-dim np.ndarray to scalar
            for col in df.columns:
                df[col] = df[col].apply(
                    lambda x: x.item() if isinstance(x, np.ndarray) else x
                )

            # Compute mean row
            feature_cols = [c for c in df.columns if c != "subject_id"]
            mean_row = pd.DataFrame(
                [
                    {
                        **{"subject_id": "mean"},
                        **df[feature_cols].mean().round(4).to_dict(),
                    }
                ]
            )
            df = pd.concat([df, mean_row], ignore_index=True)

            dfs_by_channel[channel] = df

        # Write to Excel sheets
        with pd.ExcelWriter(outfile) as writer:
            for channel, df in dfs_by_channel.items():
                df.to_excel(writer, sheet_name=channel.replace("/", "_"), index=False)
