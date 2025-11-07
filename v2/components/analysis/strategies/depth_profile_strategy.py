import os
import logging
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import json

from v2.components.analysis.strategy_base import BaseStrategy


class DepthProfileStrategy(BaseStrategy):
    """
    Strategy for drawing depth profiles of a dataset
    """

    def initialize(self):
        self.output_dir = self.output_dir / Path(self.__class__.__name__)
        os.makedirs(self.output_dir, exist_ok=True)

    def process(self):

        logging.info(f"Performing analysis: {self.__class__.__name__}")

        for i, target_chromo in enumerate(self.config.target_channels):

            logging.info(f"{target_chromo}.")

            for target_feature in self.config.target_features:

                logging.info(f"{target_feature}.")

                if "unlabelled" in self.config.profile_labels:
                    # draw depth profile for all subjects, without any labels
                    x_coordinates, z_coordinates, feature_intensities = (
                        self._load_profile_data(
                            self.subjects, target_chromo, target_feature
                        )
                    )
                    self._plot_profile(
                        [[x_coordinates], [z_coordinates]],
                        [feature_intensities],
                        ["All Subjects"],
                        ["blue"],
                        target_feature,
                        target_chromo,
                        self.output_dir,
                        plot_trendline=self.config.plot_trendline,
                        suffix="Unlabelled",
                    )

                if "GT" in self.config.profile_labels:
                    # draw depth profile for all subjects, with labelled by group id (no trendline)

                    feature_intensities_list = []
                    x_positions_list = []
                    z_positions_list = []

                    for group_id in self.config.gt_groups:

                        # get subjects in self.subjects that belong to the current group_id
                        subject_group = [
                            subject
                            for subject in self.subjects
                            if subject.group_id == group_id
                        ]

                        if not subject_group:
                            logging.warning(
                                f"No subjects found for group_id {group_id} in channel {target_chromo}."
                            )
                            continue

                        x_coordinates, z_coordinates, feature_intensities = (
                            self._load_profile_data(
                                subject_group, target_chromo, target_feature
                            )
                        )
                        feature_intensities_list.append(feature_intensities)
                        x_positions_list.append(x_coordinates)
                        z_positions_list.append(z_coordinates)

                    self._plot_profile(
                        [x_positions_list, z_positions_list],
                        feature_intensities_list,
                        self.config.gt_group_labels,
                        self.config.gt_colors,
                        target_feature,
                        target_chromo,
                        self.output_dir,
                        plot_trendline=False,
                        suffix="GT_Labelled",
                    )

                if "CM" in self.config.profile_labels:
                    # draw depth profile for all subjects, with labelled by confusion matrix (requires classification metrics)
                    assert (
                        len(self.config.cm_group_labels)
                        == len(self.config.cm_colors)
                        == 4
                    ), "cm_group_labels requires 4 group labels and colors (tp, tn, fp, fn)."

                    (
                        tp_subject_ids,
                        tn_subject_ids,
                        fp_subject_ids,
                        fn_subject_ids,
                        cutoff,
                        roc_auc,
                    ) = self._load_cl_results(target_chromo, target_feature)

                    intensities_list = []
                    x_positions_list = []
                    z_positions_list = []

                    for subject_group_ids in [
                        tp_subject_ids,
                        tn_subject_ids,
                        fp_subject_ids,
                        fn_subject_ids,
                    ]:

                        # get subjects in self.subjects that belong to the current group
                        subject_group = [
                            subject
                            for subject in self.subjects
                            if subject.subject_id in subject_group_ids
                        ]
                        if not subject_group:
                            logging.warning(
                                f"No subjects found for group {subject_group_ids} in channel {target_chromo}."
                            )
                            # continue

                        x_coordinates, z_coordinates, feature_intensities = (
                            self._load_profile_data(
                                subject_group, target_chromo, target_feature
                            )
                        )
                        intensities_list.append(feature_intensities)
                        x_positions_list.append(x_coordinates)
                        z_positions_list.append(z_coordinates)

                    self._plot_profile(
                        [x_positions_list, z_positions_list],
                        intensities_list,
                        self.config.cm_group_labels,
                        self.config.cm_colors,
                        target_feature,
                        target_chromo,
                        self.output_dir,
                        plot_trendline=False,
                        cutoff=cutoff,
                        suffix="CM_Labelled",
                    )

    def _load_profile_data(self, subjects, chromo, feature):
        """
        Load the profile data for the given subjects, chromo, and feature.
        Returns a list of intensities and positions.
        """

        feature_intensities = self.group_feature_intensities(subjects, chromo, feature)

        x_coordinates = self.group_feature_intensities(
            subjects, chromo, "roi-centroid-x"
        )
        z_coordinates = self.group_feature_intensities(
            subjects, chromo, "roi-centroid-z"
        )
        return x_coordinates, z_coordinates, feature_intensities

    def _load_cl_results(self, chromo, feature):
        """
        Load the classification results for the given chromo and feature.
        Returns subject_ids for true positives, true negatives, false positives, and false negatives,
        along with the cutoff value and roc_auc
        Requires the classification results to be in the analysis directory
        """
        cl_file_path = os.path.join(
            self.output_dir.parent,
            "GroupComparisonStrategy",
            f"cl_metrics_{chromo.replace('/', '_')}_{feature.split('_')[-1]}.json",
        )

        if not os.path.exists(cl_file_path):
            raise FileNotFoundError(
                f"Classification results file {cl_file_path} not found. Run a classification task (in GroupComparisonStrategy) first."
            )

        with open(cl_file_path, "r") as f:
            cl_results = json.load(f)

        return (
            cl_results["confusion_matrix"]["true_positive"],
            cl_results["confusion_matrix"]["true_negative"],
            cl_results["confusion_matrix"]["false_positive"],
            cl_results["confusion_matrix"]["false_negative"],
            cl_results["cutoff"],
            cl_results["roc_auc"],
        )

    @staticmethod
    def _plot_profile(
        positions_list,
        intensities_list,
        labels,
        colors,
        feature,
        chromo,
        output_dir,
        plot_trendline=True,
        cutoff=None,
        suffix="",
        x_lims=[(0.015, 0.03), (0.015, 0.03)],
    ):

        assert len(intensities_list) == len(
            labels
        ), "Length of intensities_list, and labels must be the same"

        # create scatterplots for all the subjects
        for i, (coord, coord_positions) in enumerate(zip(["X", "Z"], positions_list)):

            logging.info(
                f"Creating depth profile plot: [Axis: {coord}; Chromo: {chromo}; Intensity Extraction: {feature}]"
            )

            coord_output_dir = os.path.join(output_dir, coord.lower())
            os.makedirs(coord_output_dir, exist_ok=True)

            # extract the coordinate positions for all subjects
            # coord_positions_list = [
            #     [pos[i] for pos in positions] for positions in positions_list
            # ]

            plt.figure()

            for coord_group_positions, group_intensities, label, color in zip(
                coord_positions, intensities_list, labels, colors
            ):
                plt.scatter(
                    coord_group_positions,
                    group_intensities,
                    label=label,
                    color=color,
                )

            # Linear regression to get trendline for all lists, degree=1 for linear
            # concatenate all lists in positions_list and intensities_list
            # all_positions = np.concatenate(coord_positions_list)
            all_intensities = np.concatenate(intensities_list)
            coord_positions = np.concatenate(coord_positions)

            # Calculate correlation and p-value
            pearson_corr, pearson_p = pearsonr(coord_positions, all_intensities)

            logging.info(
                f"Pearson correlation (r) between {coord} coordinate and {feature} intensity: {pearson_corr:.4f} (p-value: {pearson_p:.4e})"
            )

            if plot_trendline:
                slope, intercept = np.polyfit(coord_positions, all_intensities, 1)
                trendline = slope * coord_positions + intercept

                # Plot trendline
                plt.plot(coord_positions, trendline, color="gray", linestyle=":")

            if cutoff is not None:
                # dotted line at y=cutoff
                plt.axhline(y=cutoff, color="lightgrey", linestyle="--", label="Cutoff")

            # Set labels, title, and limits
            # plt.xlim(-0.0045, 0.0075)
            plt.xlim(x_lims[i])
            plt.xlabel(f"ROI {coord} Coordinate (m)")
            plt.ylabel(f"MSOT ROI Int. ({feature})")
            plt.title(
                f"{chromo}: ROI {coord} Coord. vs Intensities {suffix} (r: {pearson_corr:.2f}, p: {pearson_p:.2f})"
            )

            # Save and close the plot

            outfile = os.path.join(
                coord_output_dir,
                f"coords_vs_intensities_{chromo.replace('/', '_')}_{suffix}.png",
            )
            plt.legend()

            plt.savefig(outfile)
            plt.close()

            logging.info(f"Saved depth profile plot to {outfile}")
