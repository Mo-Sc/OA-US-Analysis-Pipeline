from v2.components.analysis.strategies.group_comparison_strategy import (
    GroupComparisonStrategy,
)
from v2.components.analysis.strategies.depth_profile_strategy import (
    DepthProfileStrategy,
)
from v2.components.analysis.strategies.iqm_strategy import (
    ImageQualityMetricsStrategy,
)


def create_analysis(strategy_name: str):
    if strategy_name == "group_comparison":
        return GroupComparisonStrategy()
    elif strategy_name == "depth_profile":
        return DepthProfileStrategy()
    elif strategy_name == "image_quality_metrics":
        return ImageQualityMetricsStrategy()
    else:
        raise ValueError(f"Unknown analysis strategy: {strategy_name}")
