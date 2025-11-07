import os
import numpy as np
from tqdm import tqdm
import h5py
import matplotlib.pyplot as plt
import logging
from v2.components.pipeline_component import PipelineComponent
from v2.data.tags import HDF5Tags


class SummaryPlot(PipelineComponent):
    """
    Creates a PNG containing pipeline images for each subject.
    """

    def _default_sources(self):
        return {}

    def _default_targets(self):
        return {}

    def _run_component(self):

        # size of a single plot (to inches)
        # should be default 100 px = 1 inch at 100 dpi
        plot_size = self.config.plot_size * 1 / plt.rcParams["figure.dpi"]

        pbar = tqdm(total=len(self.subjects), desc="Plotting Summary")

        for subject in self.subjects:
            pbar.set_description(
                f"Plotting summary for Study {subject.study_id} Scan {subject.scan_id} Frame {subject.frame_id}"
            )

            data = {}

            # for calculating grid size
            max_groups = 0
            max_datasets = 0

            # hdf5 file is not loaded using subject.load_dataset for efficiency
            # as multiple datasets are loaded
            with h5py.File(subject.hdf5_path, "r") as f:
                hdf5_groups = list(f.keys())

                for group in self.config.plotable_groups:
                    if group not in hdf5_groups:
                        logging.warning(
                            f"Group {group} not found in subject {subject.subject_id}, skipping."
                        )
                        continue

                    data[group] = {}
                    datasets = list(f[group].keys())

                    # reverse datasets to have US first (minor visual preference)
                    for dataset in datasets[::-1]:
                        # only load plottable datasets
                        if dataset in [
                            HDF5Tags.OA,
                            HDF5Tags.US,
                            HDF5Tags.SEG,
                        ]:
                            hdf5_data = f[group][dataset]
                            channel_names = self.hdf5_attr_list(
                                hdf5_data.attrs["channel_names"]
                            )
                            # flipud manually and add together with channel names
                            data[group][dataset] = (
                                np.flip(hdf5_data[()], 1),
                                channel_names,
                            )

                    max_datasets = max(max_datasets, len(data[group]))
            max_groups = max(max_groups, len(data))

            # get the channel index for the OA plot
            # channel_id = subject.metadata["oa_channel_names"].index(
            #     self.config.channel_name
            # )
            # channel_id = metadata["channel_names"].index(self.config.channel_name)

            # Create figure with subplots
            fig, axes = plt.subplots(
                max_datasets,
                max_groups,
                figsize=(plot_size * max_groups, plot_size * max_datasets),
            )
            if max_groups == 1 and max_datasets == 1:
                axes = np.array([[axes]])
            elif max_groups == 1 or max_datasets == 1:
                axes = axes.reshape((max_datasets, max_groups))
            axes = axes.flatten()
            # ax_idx = 0
            for col, group in enumerate(self.config.plotable_groups):
                for row in range(max_datasets):
                    ax = axes[row * max_groups + col]
                    if group not in data or row >= len(data[group]):
                        ax.axis("off")
                        continue
                    dataset = list(data[group].keys())[row]
                    array, channel_names = data[group][dataset]
                    if dataset == "OA":
                        channel_id = channel_names.index(self.config.oa_channel_name)
                    else:
                        channel_id = 0
                    if dataset == HDF5Tags.OA:
                        ax.imshow(array[channel_id], cmap="viridis")
                        ax.set_title(f"{group} - {dataset}")
                    elif dataset == HDF5Tags.US:
                        ax.imshow(array[0], cmap="gray")
                        ax.set_title(f"{group} - {dataset}")
                    elif dataset == HDF5Tags.SEG:
                        ax.imshow(array[0], cmap="jet")
                        ax.set_title(f"{group} - {dataset}")
                    else:
                        ax.imshow(array, cmap="gray")
                        ax.set_title(f"{group} - {dataset}")

                    ax.axis("off")

            fig.suptitle(
                f"Group: {subject.group_id} | Study: {subject.study_id} | Scan: {subject.scan_id} | Frame: {subject.frame_id} | OA Channel: {self.config.oa_channel_name}",
                fontsize=16,
                y=1.02,
            )
            plt.tight_layout()
            output_path = os.path.join(
                self.output_dir,
                f"{subject.subject_id}_summary.png",
            )
            plt.savefig(output_path, bbox_inches="tight")
            plt.close(fig)

            pbar.update(1)

        pbar.close()
