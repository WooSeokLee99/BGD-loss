## Boundary Gaussian distance loss for computer vision model

Paper: Boundary Gaussian Distance Loss Function for Enhancing Character Extraction from High-Resolution Scans of Ancient Metal-Type Printed Books<br>
<br>
DOI: https://doi.org/10.3390/electronics13101957<br>
<br>
To address the challenges of poor-quality raw data and imperfect GT labels, we propose a boundary-aware loss function based on Gaussian distance. 
The proposed loss function encourages the model to produce smooth and structurally consistent boundaries by comparing the boundary regions of the prediction and GT after Gaussian smoothing.<br>
<br>
Instead of directly computing pixel-wise distances between boundary points, the method approximate distance value through Gaussian-blurred boundary maps, which stabilizes training under noisy supervision and degraded document conditions.


## Challenge - Raw Data

<img src="assets/raw_data.png" width="70%" />
<br>
Unlike Gutenberg's printing, traditional Korean printing relied on manual pressing techniques using wooden sticks or cotton pads. 
As a result, the printing quality is often inconsistent, leading to various degradation patterns in scanned documents.<br>
<br>
Common defects include:<br>
<br>
● Holes inside characters caused by uneven ink distribution (a-upper)<br>
● Ink spreading or splashing outside character regions<br>
● Irregular strokes caused by ununiform printing pressure<br>
● Ink diffusion along paper fibers that were not fully processed during traditional paper manufacturing (a-lower)<br>
<br>
Due to these factors, character boundaries are often ambiguous, and existing segmentation networks struggle to effectively suppress noise such as broken strokes, internal holes, and background stains, resulting in unsatisfactory boundary quality.


## Challenge - Labeling

<img src="assets/labeling.png" width="70%" />
Low-quality printing also reduce the reliability of ground truth(GT).
Since GT masks are manually refined from model-generated pre-segmentation results, pixel-level inaccuracies are unavoidable, especially when dealing with high-resolution historical documents.<br>
<br>
In regions where the boundary between foreground and background is ambiguous, small labeling errors frequently remain, which introduces noise into the training process and limits the performance of boundary-sensitive segmentation models.


## Proposed Loss Function



## Outlier Robustness



## Comparision with Hausdorff Distance Loss


## Quantitative Evaluation



## Qualitative Evaluation


## Generalization
