import logging
import os
import json
from tqdm import tqdm
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from skimage.metrics import (
    mean_squared_error,
    peak_signal_noise_ratio,
)
from v2.components.analysis.strategy_base import BaseStrategy
from v2.components.analysis.metrics import masked_ssim
from v2 import attr_to_list


class ImageQualityMetricsStrategy(BaseStrategy):

    def initialize(self):
        """Initialize output directory."""
        self.output_dir = self.output_dir / Path(self.__class__.__name__)
        os.makedirs(self.output_dir, exist_ok=True)

        # sanity check
        if (
            self.config.recon_group == self.config.gt_group
            and self.config.recon_dataset == self.config.gt_dataset
        ):
            logging.warning("Reconstruction and ground truth datasets are the same?")

    def process(self):
        """Compute MSE, PSNR, SSIM and optionally export difference images."""
        logging.info(f"Performing analysis: {self.__class__.__name__}")

        results = []
        pbar = tqdm(total=len(self.subjects), desc="Postprocessing")

        for subject in self.subjects:
            pbar.set_description(
                f"Calculating Image Metrics, Study {subject.study_id} Scan {subject.scan_id} Frame {subject.frame_id}"
            )

            recon_data, recon_meta = subject.load_dataset(
                self.config.recon_group, self.config.recon_dataset
            )
            gt_data, gt_meta = subject.load_dataset(
                self.config.gt_group, self.config.gt_dataset
            )

            # load roi if specified
            if self.config.roi_group and self.config.roi_dataset:
                roi_data, _ = subject.load_dataset(
                    self.config.roi_group, self.config.roi_dataset
                )
                # leading dim was just for consistency
                roi_img = roi_data[0]
                # binarize to where roi is self.config.roi_target_class_id
                roi_img = roi_img == self.config.roi_target_class_id
            else:
                roi_img = None

            try:
                recon_channel_names = attr_to_list(recon_meta["channel_names"])
                gt_channel_names = attr_to_list(gt_meta["channel_names"])
            except KeyError:
                # this uses hardcoded fallback for known channel configs
                # necessary for old datasets without channel names in metadata
                # TODO: remove this fallback in future versions
                logging.warning(
                    f"Channel names not found in metadata for {subject.subject_id}. Falling back to hardcoded channel names."
                )
                from v2.utils.channels import infer_channel_names_from_shape

                gt_channel_names = infer_channel_names_from_shape(gt_data.shape)
                recon_channel_names = infer_channel_names_from_shape(recon_data.shape)

                logging.warning(f"Inferred GT channels: {gt_channel_names}")
                logging.warning(f"Inferred Recon channels: {recon_channel_names}")

            # TODO: remove this check, but for now keep it for debugging
            assert (
                recon_channel_names
                == gt_channel_names
                == [
                    "700",
                    "850",
                    "980",
                ]
            ), f"DEBUGGING NOTE: Expected channels ['700', '850', '980'], but got recon: {recon_channel_names}, gt: {gt_channel_names}"

            subject_result = {"subject_id": subject.subject_id, "channels": {}}

            for target_channel in self.config.target_channel_names:

                recon_img = recon_data[recon_channel_names.index(target_channel)]
                gt_img = gt_data[gt_channel_names.index(target_channel)]
                assert recon_img.shape == gt_img.shape

                # Apply ROI mask if provided, otherwise flatten whole image
                if roi_img is not None:
                    assert roi_img.shape == recon_img.shape
                    mask = roi_img
                else:
                    mask = np.ones_like(gt_img, dtype=bool)

                # Flatten masked pixels
                gt_1d = gt_img[mask]
                recon_1d = recon_img[mask]

                # Normalize if required
                if self.config.normalize:
                    gt_1d = self._normalize_min_max(gt_1d)
                    recon_1d = self._normalize_min_max(recon_1d)

                channel_metrics = {}

                # Compute metrics on masked/flattened arrays
                if "MSE" in self.config.metrics:
                    channel_metrics["MSE"] = mean_squared_error(gt_1d, recon_1d)

                if "PSNR" in self.config.metrics:
                    channel_metrics["PSNR"] = peak_signal_noise_ratio(
                        recon_1d, gt_1d, data_range=1.0
                    )

                # For SSIM and DIFF, reconstruct full 2D arrays only once
                if "SSIM" in self.config.metrics or "DIFF" in self.config.metrics:
                    gt_full_norm = np.zeros_like(gt_img, dtype=float)
                    recon_full_norm = np.zeros_like(recon_img, dtype=float)
                    gt_full_norm[mask] = gt_1d
                    recon_full_norm[mask] = recon_1d

                if "SSIM" in self.config.metrics:
                    # SSIM has to be computed on full images
                    channel_metrics["SSIM"] = masked_ssim(
                        gt_full_norm, recon_full_norm, mask, data_range=1.0
                    )

                    # OPTIONAL: SSIM with different window sizes for parameter tuning
                    # channel_metrics["SSIM_11"] = masked_ssim(
                    #     gt_full_norm, recon_full_norm, mask, data_range=1.0, win_size=11
                    # )
                    # channel_metrics["SSIM_21"] = masked_ssim(
                    #     gt_full_norm, recon_full_norm, mask, data_range=1.0, win_size=21
                    # )
                    # channel_metrics["SSIM_51"] = masked_ssim(
                    #     gt_full_norm, recon_full_norm, mask, data_range=1.0, win_size=51
                    # )

                if "DIFF" in self.config.metrics:
                    self._save_diff_image(
                        gt_full_norm,
                        recon_full_norm,
                        subject.subject_id,
                        target_channel,
                        channel_metrics,
                    )

                subject_result["channels"][str(target_channel)] = channel_metrics

            results.append(subject_result)

            pbar.update(1)
        pbar.close()

        # Mean metrics
        mean_metrics = {}
        for ch in self.config.target_channel_names:
            mean_metrics[str(ch)] = {
                metric: np.mean(
                    [r["channels"][str(ch)].get(metric, np.nan) for r in results]
                )
                for metric in [
                    "MSE",
                    "PSNR",
                    "SSIM",
                    # "SSIM_11",
                    # "SSIM_21",
                    # "SSIM_51",
                ]
                if metric in self.config.metrics
                # or metric
                # in [
                #     "SSIM_11",
                #     "SSIM_21",
                #     "SSIM_51",
                # ]
            }

        logging.info(f"Mean Metrics per channel: {json.dumps(mean_metrics, indent=2)}")

        # add mean metrics to results
        results.append({"mean_metrics": mean_metrics})

        # Save results
        outfile = self.output_dir / "image_quality_metrics.json"
        with open(outfile, "w") as f:
            json.dump(results, f, indent=4)

        logging.info(f"Metrics saved to {outfile}")

    def _save_diff_image(self, gt, recon, subject_id, channel, channel_metrics):
        """Save a PNG with GT, reconstruction, and difference image, including color bars."""
        diff = np.abs(gt - recon)

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        im0 = axes[0].imshow(gt, cmap="gray")
        axes[0].set_title("Ground Truth")
        axes[0].axis("off")
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        im1 = axes[1].imshow(recon, cmap="gray")
        axes[1].set_title("Reconstruction")
        axes[1].axis("off")
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        im2 = axes[2].imshow(diff, cmap="hot")
        axes[2].set_title("Difference")
        axes[2].axis("off")
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        # Dynamically build metrics string for the title
        metrics_str = ", ".join(
            f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
            for k, v in channel_metrics.items()
        )
        plt.suptitle(f"Subject: {subject_id}, Channel: {channel}\n{metrics_str}")

        out_path = self.output_dir / "img_diffs" / f"diff_{subject_id}_{channel}.png"
        os.makedirs(out_path.parent, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close(fig)

    @staticmethod
    def _normalize_min_max(img):
        """Min-max normalize array to [0, 1]."""
        img_min, img_max = np.min(img), np.max(img)
        return (
            np.zeros_like(img)
            if img_max - img_min == 0
            else (img - img_min) / (img_max - img_min)
        )

    @staticmethod
    def _normalize_zscore(img):
        """Z-score normalize array to mean 0 and std 1."""
        img_mean, img_std = np.mean(img), np.std(img)
        return np.zeros_like(img) if img_std == 0 else (img - img_mean) / img_std
