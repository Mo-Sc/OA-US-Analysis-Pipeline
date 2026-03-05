# Automatic Muscle Segmentation for the Diagnosis of Peripheral Artery Disease Using Multispectral Optoacoustic Tomography

This README contains some visualizations from the previous version, used in [https://doi.org/10.1117/12.3049067](https://doi.org/10.1117/12.3049067).

**Abstract:**

*Multispectral optoacoustic tomography (MSOT) is an imaging modality that visualizes chromophore concentrations, such as oxygenated and deoxygenated hemoglobin, aiding in the diagnosis of blood-perfusion-related
diseases like peripheral artery disease (PAD). Previous MSOT-based diagnostic studies involved experts manually
selecting a region of interest (ROI) with a predefined shape in the target muscle to analyze blood oxygenation.
This study automates this process using a deep-learning-based segmentation model applied to co-registered
ultrasound images.*

*Our pipeline automatically generates an ROI and places it in the MSOT image by segmenting the target
muscle in the ultrasound image. We evaluated its performance using two PAD-related datasets. Our automati-
cally generated ROIs achieved areas under the ROC curve (AUCs) of 0.87 and 0.76 at classifying PAD patients,
comparable to manually drawn ROIs by clinical experts. This approach could reduce annotation effort in future
MSOT studies while providing ROIs with greater physiological relevance.*

  
## Visualizations

<img src="misc/figures/pp_overview.png" alt="Pipeline Overview" width="80%"/>

### Segmentation Results

<img src="misc/figures/seg_results.png" alt="Segmentation Results" width="80%"/>

### Sample ROIs based on Ultrasound

<table>
  <tr>
    <td align="center">
      <p><strong>Ellipse</strong></p>
      <img src="misc/figures/elliptic_roi.png" alt="Elliptic ROI" width="90%"/>
    </td>
    <td align="center">
      <p><strong>Complex</strong></p>
      <img src="misc/figures/polygon_roi.png" alt="Polygon ROI" width="90%"/>
    </td>
  </tr>
</table>


### ROIs and Optoacoustic

<img src="misc/figures/msot_ic_2_samples.png" alt="Samples from Dataset 2" width="80%"/>


