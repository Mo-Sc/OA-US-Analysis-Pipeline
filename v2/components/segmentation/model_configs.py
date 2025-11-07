from dataclasses import dataclass, field
from v2.configs import Config
from typing import List, Dict


DEFAULT_LABELS = {
    "background": 0,
    "Haut": 1,
    "Faszie1": 2,
    "Muskel1": 3,
    "Muskel2": 4,
    "Membran": 5,
    "Faszie2": 6,
    "Vorlaufstrecke": 7,
    "SAT": 8,
    "Gel": 9,
}
DEFAULT_COLORS = {
    "background": "black",
    "Haut": "magenta",
    "Faszie1": "red",
    "Muskel1": "green",
    "Muskel2": "blue",
    "Membran": "cyan",
    "Faszie2": "orange",
    "Vorlaufstrecke": "gray",
    "SAT": "purple",
    "Gel": "pink",
}

# --- MSOT Dataset Mean and Std values ---

DS_MEAN = -9.541781183432969
DS_STD = 12.72830725961859

# Pretrained models
MODEL_PATH = ""


@dataclass
class cUNetConfig(Config):
    mode: str = "batch"
    model_weights: str = MODEL_PATH
    img_size: int = 224
    ds_mean: float = DS_MEAN
    ds_std: float = DS_STD
    in_channels: int = 1
    n_classes: int = 10
    patch_size: List[int] = field(default_factory=lambda: [224, 224])
    batch_size: int = 4
    features_per_stage: List[int] = field(
        default_factory=lambda: [32, 64, 128, 256, 512, 512]
    )
    kernel_sizes: List[List[int]] = field(default_factory=lambda: [[3, 3]] * 6)
    strides: List[List[int]] = field(
        default_factory=lambda: [[1, 1], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]]
    )
    class_dict: Dict = field(default_factory=lambda: DEFAULT_LABELS)
    color_dict: Dict = field(default_factory=lambda: DEFAULT_COLORS)

    png_export: bool = True
