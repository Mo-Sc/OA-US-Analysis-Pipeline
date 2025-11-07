from v2 import StepNames
from v2.components.preprocessing.preprocessing_component import PreprocessingComponent
from v2.components.segmentation.segmentation_component import SegmentationComponent
from v2.components.postprocessing.postprocessing_component import (
    PostprocessingComponent,
)
from v2.components.roi_placement.roi_placement_component import ROIPlacementComponent
from v2.components.extraction.extraction_component import ExtractionComponent
from v2.components.analysis.analysis_component import AnalysisComponent
from v2.utils.summary_plot import SummaryPlot
from v2.configs import *
from v2.components.segmentation.model_configs import cUNetConfig
from v2.components.analysis.strategy_configs import GroupComparisonConfig

from v2.data.load_subjects import collect_subjects_demo

# global pipeline config
global_config = GlobalConfig(
    output_dir="datav2/output",
    overwrite=False,
    run_id=None,
)

# dataset config
data_config = DataConfig(
    input_dir="seg-cl-pipeline/misc/demo",
    us_channel_names=["US"],
    oa_channel_names=["Hb", "HbO2"],
)

# configure segmentation step (default config)
segmentation_config = SegmentationConfig(arch="cUNet", model_config=cUNetConfig())

# config for the downstream analysis (comparing the two groups)
group_comp_config_demo = GroupComparisonConfig(
    groups=[0, 1],
    group_labels=["PAD", "HV"],
    combine_studies=False,
    target_channels=["Hb", "HbO2"],
    stat_test="ttest_student",
    classification_metrics=True,
    invert=[True, False],
)
analysis_config = AnalysisConfig(
    strategy_name="group_comparison", strategy_config=group_comp_config_demo
)

# load the subjects in the dataset
subjects = collect_subjects_demo(
    data_config=data_config,
    file_ending=".npy",
)


pipeline = [
    PreprocessingComponent(
        name=StepNames.PREPROCESSING,
        config=PreprocessingConfig(),
        subjects=subjects,
    ),
    SegmentationComponent(
        name=StepNames.SEGMENTATION,
        config=segmentation_config,
        subjects=subjects,
    ),
    PostprocessingComponent(
        name=StepNames.POSTPROCESSING,
        config=PostprocessingConfig(),
        subjects=subjects,
    ),
    ROIPlacementComponent(
        name=StepNames.ROI_PLACEMENT,
        config=ROIPlacementConfig(),
        subjects=subjects,
    ),
    ExtractionComponent(
        name=StepNames.EXTRACTION,
        config=ExtractionConfig(),
        subjects=subjects,
    ),
    AnalysisComponent(
        name=StepNames.ANALYSIS,
        config=analysis_config,
        subjects=subjects,
    ),
    SummaryPlot(
        name=StepNames.SUMMARY_PLOT,
        config=SummaryPlotConfig(),
        subjects=subjects,
    ),
]
