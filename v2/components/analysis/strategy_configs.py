from dataclasses import dataclass, field
from v2.configs import Config
from typing import List
from v2.data.tags import HDF5Tags


@dataclass
class GroupComparisonConfig(Config):
    """
    Configuration for group comparison analysis.
    Creates boxplot comparison between the different groups
    Creates scatterplot with trendline
    Calculates classification metrics and plots ROC curves
    Performs statistical tests
    """

    groups: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6])
    group_labels: List[str] = field(
        default_factory=lambda: [
            "Type I",
            "Type II",
            "Type III",
            "Type IV",
            "Type V",
            "Type VI",
        ]
    )

    target_features: List[str] = field(
        default_factory=lambda: [
            "original_firstorder_Mean",
            "original_firstorder_Median",
        ]
    )
    target_channels: List[str] = field(default_factory=lambda: ["Hb", "HbO2", "mSO2"])

    combine_studies: bool = (
        True  # combine subjects with same study id by taking the mean of the features
    )

    stat_test: str = field(default="ttest_student")  # or 'anova', 'spearman'
    line_type: str = field(default="lowess")  # or 'lowess', 'linear'
    fig_size: tuple = field(default=(6, 5))

    classification_metrics: bool = False
    invert: List[bool] = field(
        default_factory=lambda: [True, False, False]
    )  # invert groups (only for classification metrics).


@dataclass
class DepthProfileConfig(Config):
    """
    Configuration for depth profile analysis.
    Draws depth profiles of a dataset.
    """

    target_channels: List[str] = field(default_factory=lambda: ["Hb", "HbO2", "mSO2"])
    target_features: List[str] = field(
        default_factory=lambda: [
            "original_firstorder_Mean",
        ]
    )
    plot_trendline: bool = True
    profile_labels: List[str] = field(
        default_factory=lambda: [
            "unlabelled",  # no labels
            "CM",  # color is confusion matrix (requires classification metrics)
            "GT",  # color is ground truth (requires group id)
        ]
    )
    # only required for GT profile label
    gt_groups: List[int] = field(default_factory=lambda: [0, 1])
    gt_group_labels: List[str] = field(
        default_factory=lambda: [
            "PAD",
            "HV",
        ]
    )
    gt_colors: List[str] = field(
        default_factory=lambda: [
            "red",
            "green",
        ]
    )

    # only required for CM profile label
    cm_group_labels: List[str] = field(
        default_factory=lambda: [
            "True Positives",
            "True Negatives",
            "False Positives",
            "False Negatives",
        ],
    )
    cm_colors: List[str] = field(
        default_factory=lambda: ["chartreuse", "olivedrab", "brown", "orange"],
    )


@dataclass
class ImageQualityMetricsConfig(Config):
    """
    Configuration for image quality metrics analysis.
    Computes MSE, PSNR, SSIM for each subject.
    If DIFF is included in metrics, saves a PNG with GT, reconstruction, and difference image.
    """

    # these overwrite the default values in AnalsisComponent
    recon_group: str = "SIMULATIONS"
    recon_dataset: str = HDF5Tags.OA
    gt_group: str = HDF5Tags.RAW
    gt_dataset: str = HDF5Tags.OA

    # if set, metrics are only computed in the ROI
    roi_group: str = HDF5Tags.ROI
    roi_dataset: str = HDF5Tags.SEG
    roi_target_class_id: int = 2

    target_channel_names: List[str] = field(
        default_factory=lambda: ["Hb", "HbO2", "mSO2"]
    )

    # Can be "MSE", "PSNR", "SSIM", "DIFF"
    metrics: List[str] = field(default_factory=lambda: ["MSE", "PSNR", "SSIM"])

    # TODO: this normalization can only do image wise min-max norm. also maybe better in preprocessing
    normalize: bool = False
