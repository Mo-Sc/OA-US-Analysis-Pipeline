import ast
from abc import ABC, abstractmethod


class BaseStrategy(ABC):
    def __init__(self):
        self.name = None
        self.config = None
        self.subjects = None
        self.sources = None
        self.output_dir = None

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def process(self):
        pass

    def group_subjects(self, group_id: int):
        """
        Returns a list of subjects with the given group_id
        """
        return [subject for subject in self.subjects if subject.group_id == group_id]

    def group_intensities(self, subject_group):
        """
        Returns a list of the extracted intensities for a given group of subjects
        """
        return [
            subject.extraction_results(
                self.sources["features"][0], self.sources["features"][1]
            )
            for subject in subject_group
        ]

    def group_feature_intensities(self, subject_group, target_chromo, target_feature):
        """
        Returns a list of the extracted intensities of a target_chromophore and a target_feature for a given group of subjects
        """
        feature_intensities = []
        for subject in subject_group:
            res_data, res_attrs = subject.extraction_results(
                self.sources["features"][0], self.sources["features"][1]
            )

            # find index of target feature in dataset
            avail_features = ast.literal_eval(res_attrs["features"])
            target_feature_id = avail_features.index(target_feature)

            # find index of target chromophore in dataset
            avail_chromos = ast.literal_eval(res_attrs["channel_names"])
            target_chromo_id = avail_chromos.index(target_chromo)

            feature_intensities.append(res_data[target_chromo_id, target_feature_id])

        return feature_intensities
