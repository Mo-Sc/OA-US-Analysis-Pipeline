import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from v2.data import parse_subject_id

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


def plot_segmentation(
    input_image,
    predicted_mask,
    label_dict=None,
    color_dict=None,
    gt_mask=None,
    output_path="output.png",
):
    """
    Save input image and predicted mask to a PNG file with consistent label colors
    and a simplified legend.
    """

    if not label_dict:
        label_dict = DEFAULT_LABELS
    if not color_dict:
        color_dict = DEFAULT_COLORS

    # Create a colormap and normalize it
    colors = [color_dict[label] for label in label_dict]
    cmap = mcolors.ListedColormap(colors[: len(label_dict)])
    norm = mcolors.BoundaryNorm(range(len(label_dict) + 1), cmap.N)

    # Prepare the plot
    if gt_mask is not None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Plot the input image
    axes[0].imshow(input_image, cmap="gray")
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    # Plot the predicted mask
    im = axes[1].imshow(predicted_mask, cmap=cmap, norm=norm, interpolation="none")
    axes[1].set_title("Predicted Mask")
    axes[1].axis("off")

    if gt_mask is not None:
        # Plot the ground truth mask
        axes[2].imshow(gt_mask, cmap=cmap, norm=norm, interpolation="none")
        axes[2].set_title("Ground Truth")
        axes[2].axis("off")

    # Add the legend directly from the colormap
    cbar = fig.colorbar(im, ax=axes, location="right", fraction=0.02, pad=0.02)
    cbar.set_ticks(range(len(label_dict)))
    cbar.set_ticklabels(list(label_dict.keys()))

    group_id, study_id, scan_id, frame_id = parse_subject_id(output_path.stem)

    fig.suptitle(
        f"Group: {group_id} | Study: {study_id} | Scan: {scan_id} | Frame: {frame_id}",
        fontsize=16,
        y=1.05,
    )
    # plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
