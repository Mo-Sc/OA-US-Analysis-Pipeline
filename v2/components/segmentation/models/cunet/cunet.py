import logging
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import os
from pathlib import Path

from v2.components.segmentation.model_base import BaseSegmentationModel
from v2 import bool_env
from v2.data.tags import HDF5Tags

from .assets.dataset import HDF5InferenceDataset
from .assets.transforms import inference_transforms, reverse_inference_transforms
from .assets.architecture import load_model


class cUNet(BaseSegmentationModel):
    def initialize(self):
        self.model = load_model(self.config).cuda()
        logging.info(f"Loaded cUNet model from {self.config.model_weights}")

    def preprocess(self):
        # Done in dataloader
        pass

    def inference(self):

        infer_transform = inference_transforms(
            self.config.ds_mean, self.config.ds_std, self.config.patch_size
        )

        infer_dataset = HDF5InferenceDataset(
            subjects=self.subjects,
            src_group=self.src_group,
            src_dataset=self.src_dataset,
            transform=infer_transform,
        )

        reverse_transform = reverse_inference_transforms(
            infer_dataset.orig_size[1:], self.config.ds_mean, self.config.ds_std
        )

        hdf5_attributes = self.config.serialize()
        hdf5_attributes["channel_names"] = [HDF5Tags.SEG]

        infer_loader = DataLoader(infer_dataset, batch_size=1, shuffle=False)

        self.model.eval()

        with torch.no_grad():

            pbar = tqdm(total=len(infer_loader), desc="Inference")

            for i, image in enumerate(infer_loader):
                # leading dim because of batch

                pbar.set_description(f"Inference on {self.subjects[i].subject_id}")

                image = image.cuda()
                output = self.model(image)

                # take the first (and only) image in the batch
                pred = torch.argmax(output, dim=1)

                # reverse transform the prediction
                img_rev, pred_rev = reverse_transform(image, pred)

                # save the segmentation map to hdf5
                self.subjects[i].add_dataset(
                    self.target_group,
                    self.target_dataset,
                    pred_rev_np := pred_rev.cpu().numpy(),
                    attributes=hdf5_attributes,
                )

                # save the segmentation map to png
                if self.config.png_export:
                    self.plot(
                        input_image=img_rev[0].cpu().numpy()[0],
                        predicted_mask=pred_rev_np[0],
                        filename=f"{self.subjects[i].subject_id}.png",
                    )

                pbar.update(1)
            pbar.close()

    def postprocess(self):
        # Nothing to be done here. postprocessing for cunet is done in the inference loop.
        pass
