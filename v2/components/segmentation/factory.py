def create_model(model_name):
    """
    Select and create the segmentation model based on the configuration.
    """
    if model_name == "MedSAM":

        raise ValueError("MedSAM is not implemented in v2")

    elif model_name == "nnUNet":

        raise ValueError("nnUNet is not implemented in v2")

    elif model_name == "cUNet":

        from v2.components.segmentation.models.cunet.cunet import cUNet

        return cUNet()
    # Add more models if needed
    else:
        raise ValueError(f"Unknown model type: {model_name}")
