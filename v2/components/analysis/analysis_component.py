import logging


from v2.components.pipeline_component import PipelineComponent
from v2.components.analysis.strategy_base import BaseStrategy
from v2.components.analysis.factory import create_analysis
from v2.data.tags import HDF5Tags


class AnalysisComponent(PipelineComponent):
    """
    Performs analysis on the extracted intensities.
    """

    def _default_sources(self):
        return {
            "features": (HDF5Tags.EXTRACTED, HDF5Tags.TABULAR),
            "oa": (HDF5Tags.RAW, HDF5Tags.OA),
        }

    def _default_targets(self):
        # currently analysis doenst write to hdf5
        # return {
        #     "primary": (HDF5Tags.ANALYSIS, HDF5Tags.TABULAR),
        # }
        return {}

    def __init__(
        self, name, config, subjects, source_overrides=None, target_overrides=None
    ):
        super().__init__(
            name=name,
            config=config,
            subjects=subjects,
            source_overrides=source_overrides,
            target_overrides=target_overrides,
        )
        self.strategy: BaseStrategy = create_analysis(config.strategy_name)

        # inject runtime context
        self.strategy.name = name
        self.strategy.subjects = subjects
        self.strategy.config = config.strategy_config
        self.strategy.sources = self.sources

    def _run_component(self):

        # suppress matplotlib info messages
        logging.getLogger("matplotlib").setLevel(logging.WARNING)

        self.strategy.output_dir = self.output_dir
        self.strategy.initialize()
        self.strategy.process()
