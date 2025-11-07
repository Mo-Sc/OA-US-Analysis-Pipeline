import logging

# from abc import abstractmethod

from v2.components.pipeline_component import PipelineComponent
from v2.components.segmentation.factory import create_model
from v2.components.segmentation.model_base import BaseSegmentationModel
from v2.data.tags import HDF5Tags


class SegmentationComponent(PipelineComponent):
    """
    Segment a 2D scan.
    """

    def _default_sources(self):
        return {
            "primary": (HDF5Tags.PREPROCESSED, HDF5Tags.US),
        }

    def _default_targets(self):
        return {
            "primary": (HDF5Tags.SEGMENTED, HDF5Tags.SEG),
        }

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
        self.seg_model: BaseSegmentationModel = create_model(config.arch)

        # inject runtime context
        self.seg_model.name = name
        self.seg_model.subjects = subjects
        self.seg_model.config = config.model_config

        self.seg_model.src_group, self.seg_model.src_dataset = self.sources["primary"]
        self.seg_model.target_group, self.seg_model.target_dataset = self.targets[
            "primary"
        ]

    def _run_component(self):

        self.seg_model.output_dir = self.output_dir
        self.seg_model.initialize()

        if self.seg_model.config.mode == "instance":
            raise NotImplementedError("Instance mode is not supported.")
        elif self.seg_model.config.mode == "batch":
            self.seg_model.preprocess()
            self.seg_model.inference()
            self.seg_model.postprocess()
        else:
            raise ValueError(f"Invalid mode: {self.seg_model.config.mode}")
