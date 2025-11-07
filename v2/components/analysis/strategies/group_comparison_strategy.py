import logging
import os
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.nonparametric.smoothers_lowess import lowess

from v2.components.analysis.strategy_base import BaseStrategy
from v2.utils.statistics import compute_stat_test
from v2.components.analysis.metrics import compute_classification_metrics
from v2.data import parse_subject_id

import plotly.graph_objects as go
import plotly.express as px


class GroupComparisonStrategy(BaseStrategy):

    def initialize(self):
        self.output_dir = self.output_dir / Path(self.__class__.__name__)
        os.makedirs(self.output_dir, exist_ok=True)

    def process(self):

        logging.info(f"Performing analysis: {self.__class__.__name__}")

        assert len(self.config.groups) == len(
            self.config.group_labels
        ), "unequal number of group ids and group labels"

        chromo_feature_aucs = {}

        for i, target_chromo in enumerate(self.config.target_channels):

            logging.info(f"{target_chromo}.")

            for target_feature in self.config.target_features:

                logging.info(f"{target_feature}.")

                # create a plot with selected chromo and target
                groups_intensities = []
                groups_subject_ids = []

                for group_id, group_label in zip(
                    self.config.groups, self.config.group_labels
                ):

                    group_subjects = self.group_subjects(group_id)

                    # collect the intensities of the group for the given target feature and chromo
                    group_intensities = self.group_feature_intensities(
                        group_subjects, target_chromo, target_feature
                    )
                    group_subject_ids = [
                        subject.subject_id for subject in group_subjects
                    ]

                    if self.config.combine_studies:
                        # combine subjects with same study id by taking the mean
                        group_intensities, group_subject_ids = self.combine_study_ids(
                            group_intensities, group_subject_ids
                        )

                    groups_intensities.append(group_intensities)
                    groups_subject_ids.append(group_subject_ids)

                    # group stats
                    logging.info(
                        f"Group {group_label} (ID: {group_id}): Total Subjects: {len(group_subjects)}; Combined Subjects: {len(group_subject_ids)}"
                    )

                title = f"Group Comparison - {target_chromo} - {target_feature.split('_')[-1]}"
                self._plot_group_comparison_plotly(
                    groups_intensities,
                    self.config.group_labels,
                    groups_subject_ids,
                    title=title,
                )
                self._plot_group_comparison(
                    groups_intensities,
                    self.config.group_labels,
                    title=title,
                )

                if self.config.classification_metrics:
                    if len(self.config.groups) != 2:
                        logging.warning(
                            f"Classification can only be performed for 2 groups. Found {len(self.config.groups)}. Skipping classification."
                        )
                        self.config.classification_metrics = False
                        continue

                    assert len(self.config.target_channels) == len(
                        self.config.invert
                    ), "Number of target channels must match number of invert flags"

                    metrics = compute_classification_metrics(
                        groups_intensities,
                        groups_subject_ids,
                        invert=self.config.invert[i],
                    )

                    # Save auc for best feauture
                    chromo_feature_aucs[
                        f"{target_chromo}_{target_feature.split('_')[-1]}"
                    ] = metrics["accuracy"]

                    title = f"ROC - {target_chromo} - {target_feature.split('_')[-1]}"
                    self._plot_roc_curve(
                        metrics["fpr"],
                        metrics["tpr"],
                        metrics["roc_auc"],
                        metrics["cutoff"],
                        title,
                    )

                    logging.info(
                        f"[Classification] Accuracy: {metrics['accuracy']:.3f}"
                    )
                    logging.info(f"[Classification] F1 Score: {metrics['f1']:.3f}")
                    logging.info(f"[Classification] ROC AUC: {metrics['roc_auc']:.3f}")
                    logging.info(
                        f"[Classification] Cutoff Value: {metrics['cutoff']:.3f}"
                    )

                    # Save to JSON
                    metrics_outfile = self.output_dir / Path(
                        f"cl_metrics_{target_chromo.replace('/', '_')}_{target_feature.split('_')[-1]}.json"
                    )

                    with open(metrics_outfile, "w") as f:
                        json.dump(metrics, f, indent=4)

                    logging.info(f"[Classification] Metrics saved to {metrics_outfile}")

        if self.config.classification_metrics:
            # plot chromo feauture combo aucs
            best_chromofeature, best_auc = self._plot_feature_comparison(
                chromo_feature_aucs
            )
            logging.info(
                f"Best chromo - feature combo: {best_chromofeature}, AUC: {best_auc:.3f}"
            )

    def _plot_group_comparison(self, group_data, group_labels, title="Comparison Plot"):
        """ """

        cfg = self.config

        # hacky way to remove last group
        # group_data = group_data[:-1]
        # group_labels = group_labels[:-1]

        x_label = "Group"
        y_label = "ROI Intensity"

        if "mSO2" in title:
            y_max = 1.0
        else:
            y_max = (
                max_val := max([max(group) for group in group_data])
            ) + 0.2 * max_val

        # y_max = 1.2

        # Prepare data
        data_points = []
        for i, group in enumerate(group_data):
            for value in group:
                data_points.append(
                    {"Group": group_labels[i], "Mean Intensity": value, "GroupIndex": i}
                )
        df = pd.DataFrame(data_points)

        # Create figure
        fig, (ax_box, ax_reg) = plt.subplots(
            1, 2, figsize=(cfg.fig_size[0] * 2, cfg.fig_size[1])
        )
        fig.suptitle(title, fontsize=14)

        # ---- Boxplot ----
        sns.boxplot(data=group_data, ax=ax_box)
        ax_box.set_title("Boxplot")
        ax_box.set_xlabel(x_label)
        ax_box.set_ylabel(y_label)
        ax_box.set_xticks(range(len(group_labels)))
        ax_box.set_xticklabels(group_labels)
        ax_box.set_ylim(0, y_max)

        # ---- Statistical Test ----
        stat_test_label = compute_stat_test(cfg.stat_test, df, group_data)

        if stat_test_label:
            ax_box.text(
                0.5,
                0.95,
                stat_test_label,
                transform=ax_box.transAxes,
                ha="center",
                va="top",
                fontsize=10,
                bbox=dict(facecolor="white", alpha=0.7),
            )

        # ---- Scatter ----
        ax_reg.scatter(df["GroupIndex"], df["Mean Intensity"], alpha=0.6)
        x_vals = df["GroupIndex"].values
        y_vals = df["Mean Intensity"].values

        # ---- Trend Line ----
        if cfg.line_type == "lowess":
            x_grid = np.linspace(min(x_vals), max(x_vals), 100)
            smooth = lowess(y_vals, x_vals, frac=0.75, return_sorted=True)
            y_interp = np.interp(x_grid, smooth[:, 0], smooth[:, 1])
            ax_reg.plot(x_grid, y_interp, color="red", label="LOWESS")

            ci = True
            if ci:
                # ---- Bootstrap CI for LOWESS ----
                n_boot = 500
                boot_preds = []

                for _ in range(n_boot):
                    sample_df = df.sample(frac=1, replace=True)
                    sample_x = sample_df["GroupIndex"].values
                    sample_y = sample_df["Mean Intensity"].values

                    if len(np.unique(sample_x)) < 2:
                        continue  # skip bad resample

                    smooth = lowess(
                        sample_y, sample_x, frac=0.75, it=3, return_sorted=True
                    )
                    y_interp = np.interp(x_grid, smooth[:, 0], smooth[:, 1])
                    boot_preds.append(y_interp)

                boot_preds = np.array(boot_preds)
                ci_lower = np.percentile(boot_preds, 2.5, axis=0)
                ci_upper = np.percentile(boot_preds, 97.5, axis=0)
                ax_reg.fill_between(
                    x_grid, ci_lower, ci_upper, color="red", alpha=0.2, label="95% CI"
                )

        elif cfg.line_type == "linear":
            sns.regplot(
                x=x_vals,
                y=y_vals,
                ax=ax_reg,
                scatter=False,
                ci=95,
                line_kws={"color": "blue"},
            )
            ax_reg.legend(["Linear Regression (95% CI)"])

        ax_reg.set_title(
            "Scatterplot" + (f" ({cfg.line_type} Trendline)" if cfg.line_type else "")
        )
        ax_reg.set_xlabel(x_label)
        ax_reg.set_ylabel(y_label)
        ax_reg.set_xticks(range(len(group_labels)))
        ax_reg.set_xticklabels(group_labels)
        ax_reg.set_ylim(0, y_max)

        plt.tight_layout()

        outfile = self.output_dir / Path(
            title.replace(" ", "").replace("-", "_").replace("/", "_").lower() + ".png"
        )
        plt.savefig(outfile, dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_roc_curve(self, fpr, tpr, roc_auc, cutoff, title):
        plt.figure(figsize=self.config.fig_size)
        plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], "k--", label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.grid(True)

        outfile = self.output_dir / Path(
            title.replace(" ", "").replace("-", "_").replace("/", "_").lower() + ".png"
        )
        plt.savefig(outfile, dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_feature_comparison(self, results):
        """
        Plot a bar chart comparing the ROC-AUC values of the features.
        results: dict, dictionary containing the ROC-AUC values of the features
        """

        # Extract top 10 features based on roc auc
        top_features = sorted(results.items(), key=lambda item: item[1], reverse=True)[
            :10
        ]

        # Separate keys and values for the bar chart
        features = [item[0] for item in top_features]
        roc_auc_values = [item[1] for item in top_features]

        plt.figure(figsize=(len(features) * 1.5, self.config.fig_size[1]))
        bars = plt.bar(features, roc_auc_values, color="skyblue")

        # Add text annotations on top of the bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        plt.title("Top 10 Chromophore-Feature-Combinations by ROC-AUC", fontsize=14)
        plt.xlabel("Chromo+Features", fontsize=12)
        plt.ylabel("ROC-AUC Value", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        outfile = self.output_dir / Path("chromo-feature_comparison.png")
        plt.savefig(outfile, dpi=300, bbox_inches="tight")
        plt.close()

        return features[0], roc_auc_values[0]

    @staticmethod
    def combine_study_ids(group_intensities, group_subject_ids):
        """
        Combine subjects with same study id by taking the mean
        """
        combined_intensities = {}
        combined_subject_ids = {}

        for intensity, subject_id in zip(group_intensities, group_subject_ids):
            _, study_id, _, _ = parse_subject_id(subject_id)
            if study_id not in combined_intensities:
                combined_intensities[study_id] = []
                combined_subject_ids[study_id] = []
            combined_intensities[study_id].append(intensity)
            combined_subject_ids[study_id].append(subject_id)

        # Take the mean of the intensities
        combined_intensities = [
            np.mean(intensities) for intensities in combined_intensities.values()
        ]

        # shorten subject id to only include group and study id (todo: bit hacky)
        combined_subject_ids = [
            subject_ids[0][0:7] for subject_ids in combined_subject_ids.values()
        ]

        return combined_intensities, combined_subject_ids

    def _plot_group_comparison_plotly(
        self,
        groups_intensities,
        group_labels,
        groups_subject_ids,
        title="Comparison Plot",
        cfg=None,
    ):
        """
        Plot group comparison using Plotly (interactive).
        Hovering over scatter points shows subject_id.
        """
        x_label = "Group"
        y_label = "ROI Intensity"

        # Flatten data for DataFrame
        data_points = []
        for group_idx, (intensities, subject_ids) in enumerate(
            zip(groups_intensities, groups_subject_ids)
        ):
            for intensity, subject_id in zip(intensities, subject_ids):
                data_points.append(
                    {
                        "Group": group_labels[group_idx],
                        "GroupIndex": group_idx,
                        "ROI Intensity": intensity,
                        "SubjectID": subject_id,
                    }
                )

        df = pd.DataFrame(data_points)

        # Compute y-axis max
        y_max = df["ROI Intensity"].max() * 1.2

        # --- Boxplot ---
        fig_box = px.box(
            df,
            x="Group",
            y="ROI Intensity",
            points="all",  # show all points
            title=title + " - Boxplot",
        )
        fig_box.update_layout(yaxis=dict(range=[0, y_max]))

        # --- Scatterplot ---
        fig_scatter = go.Figure()

        # Add scatter points with subject ID as hover text
        for group_idx, group_label in enumerate(group_labels):
            group_df = df[df["GroupIndex"] == group_idx]
            fig_scatter.add_trace(
                go.Scatter(
                    x=[group_idx] * len(group_df),
                    y=group_df["ROI Intensity"],
                    mode="markers",
                    name=group_label,
                    text=group_df["SubjectID"],
                    hovertemplate="<b>Subject ID:</b> %{text}<br><b>Intensity:</b> %{y}<extra></extra>",
                    marker=dict(size=8, opacity=0.7),
                )
            )

        # Add LOWESS trendline
        x_vals = df["GroupIndex"].values
        y_vals = df["ROI Intensity"].values

        if cfg and cfg.line_type == "lowess":
            x_grid = np.linspace(x_vals.min(), x_vals.max(), 100)
            smooth = lowess(y_vals, x_vals, frac=0.75, return_sorted=True)
            y_interp = np.interp(x_grid, smooth[:, 0], smooth[:, 1])
            fig_scatter.add_trace(
                go.Scatter(
                    x=x_grid,
                    y=y_interp,
                    mode="lines",
                    name="LOWESS",
                    line=dict(color="red"),
                )
            )

        elif cfg and cfg.line_type == "linear":
            slope, intercept = np.polyfit(x_vals, y_vals, 1)
            line_y = slope * x_vals + intercept
            fig_scatter.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=line_y,
                    mode="lines",
                    name="Linear Fit",
                    line=dict(color="blue"),
                )
            )

        fig_scatter.update_layout(
            title=title + " - Scatterplot",
            xaxis=dict(
                title=x_label,
                tickvals=list(range(len(group_labels))),
                ticktext=group_labels,
            ),
            yaxis=dict(title=y_label, range=[0, y_max]),
        )

        # --- Save as HTML ---
        outfile_base = self.output_dir / Path(
            title.replace(" ", "").replace("-", "_").replace("/", "_").lower() + ".png"
        )
        fig_box.write_html(str(outfile_base) + "_box.html")
        fig_scatter.write_html(str(outfile_base) + "_scatter.html")
