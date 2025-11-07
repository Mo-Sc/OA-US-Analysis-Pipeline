from abc import ABC, abstractmethod

from v2.utils.visualizations import plot_segmentation


class BaseSegmentationModel(ABC):
    def __init__(self):
        self.name = None
        self.config = None
        self.subjects = None
        self.src_group = None
        self.src_dataset = None
        self.output_dir = None

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def preprocess(self):
        pass

    @abstractmethod
    def inference(self):
        pass

    @abstractmethod
    def postprocess(self):
        pass

    def plot(self, input_image, predicted_mask, filename):
        """
        Save input image and predicted mask to a PNG file with consistent label colors
        and legend.
        """
        plot_segmentation(
            input_image=input_image,
            predicted_mask=predicted_mask,
            label_dict=self.config.class_dict,
            color_dict=self.config.color_dict,
            output_path=self.output_dir / filename,
        )
