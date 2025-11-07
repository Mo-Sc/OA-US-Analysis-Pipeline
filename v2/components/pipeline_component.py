from abc import ABC, abstractmethod
import os
from pathlib import Path
from typing import List
import logging
from v2.configs import Config
from v2.data.subject import Subject
from v2 import StepNames, attr_to_list


class PipelineComponent(ABC):
    """
    Base class for all pipeline steps.
    Each step should implement the `run()` method to define its specific logic.
    custom_src_group can be used to override the default source group for the step.
    custom_src_dataset can be used to override the default source dataset for the step.
    """

    def __init__(
        self,
        name: StepNames,
        config: Config,
        subjects: List[Subject],
        source_overrides: dict = None,
        target_overrides: dict = None,
    ):
        self.name = name
        self.config = config
        self.subjects = subjects
        self.sources = self._default_sources()
        self.targets = self._default_targets()

        # override default source and target group and dataset
        if source_overrides:
            self.sources = source_overrides
        if target_overrides:
            self.targets = target_overrides

    def run(self):
        """
        Run the component. This method should be called to execute the step.
        It will call the `_run_component()` method defined in the subclass.
        """

        logging.info(f"--- {self.name} ---")

        self._initialize()
        self._run_component()

        logging.info(f"--- {self.name} completed ---")

    def _initialize(self):
        """
        Initialize the step.
        """

        # create output directory
        self.output_dir = Path(
            os.environ["RUN_OUTDIR"],
            "data",
            f".{str(self.name).lower()}",
        )
        os.makedirs(self.output_dir, exist_ok=True)

    @abstractmethod
    def _run_component(self):
        """
        This method should be implemented in subclasses to define the specific logic of the step.
        """
        pass

    @abstractmethod
    def _default_sources(self) -> dict:
        """
        Should return a dict of source keys and their default (group, dataset) tuples.
        Example: {"roi": (HDF5Tags.ROI, HDF5Tags.SEG), "oa": (HDF5Tags.RAW, HDF5Tags.OA)}
        """
        pass

    @abstractmethod
    def _default_targets(self) -> dict:
        """
        Should return a dict of target keys and their default (group, dataset) tuples.
        Example: {"roi": (HDF5Tags.ROI, HDF5Tags.SEG), "oa": (HDF5Tags.RAW, HDF5Tags.OA)}
        """
        pass

    @staticmethod
    def hdf5_attr_list(attr):
        """Convert stringyfied hdf5 attribute to list."""
        return attr_to_list(attr)
