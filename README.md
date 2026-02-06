## Boundary Gaussian Distance Loss

Paper: Boundary Gaussian Distance Loss Function for Enhancing Character Extraction from High-Resolution Scans of Ancient Metal-Type Printed Books<br>
<br>
DOI: https://doi.org/10.3390/electronics13101957<br>
<br>
To address the challenges of poor-quality raw data and imperfect GT labels, we propose a boundary-aware loss function based on Gaussian distance. 
The proposed loss function encourages the model to produce smooth and structurally consistent boundaries by comparing the boundary regions of the prediction and GT after Gaussian smoothing.<br>
<br>
Instead of directly computing pixel-wise distances between boundary points, the method approximate distance value through Gaussian-blurred boundary maps, which stabilizes training under noisy supervision and degraded document conditions.


## Challenge - Raw Data

<img src="assets/2.png" width="70%" />
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

<img src="assets/3.png" width="70%" />
Low-quality printing also reduce the reliability of ground truth(GT).
Since GT masks are manually refined from model-generated pre-segmentation results, pixel-level inaccuracies are unavoidable, especially when dealing with high-resolution historical documents.<br>
<br>
In regions where the boundary between foreground and background is ambiguous, small labeling errors frequently remain, which introduces noise into the training process and limits the performance of boundary-sensitive segmentation models.


## Proposed Loss Function

<img src="assets/4-1.png" width="70%" />
<img src="assets/4-2.png" width="70%" />
<img src="assets/4-3.png" width="70%" />
<br>
1. Get boundary image with Erosion. (a)<br>
2. Blur the boundary image with Gaussian kernel. (b)<br>
3. Calc M value: Average of Gaussian value on baundary position. (c,d)<br>
4. Calc difference between 4 different M.<br>
<br>
<img src="assets/4-4.png" width="70%" />
<br>
Each graph shows certical cross-section of results.<br>
The pixel value in boundary position is 1 and maximum Gaussian value has set as 1 via normalization.<br>
And we set other smaller values arbitrarily.


## Parameter

<img src="assets/5-1.png" width="70%" />
<img src="assets/5-2.png" width="70%" />
<br>
The performance of the proposed loss function is influenced by the Gaussian kernel parameter σ.<br>
Standard deviation σ controls the spatial range of boundary interaction.<br>
In character segmentation tasks, approximately half of the average stroke width provides the best trade-off between boundary smoothness and localization.


## Outlier Robustness

<img src="assets/6.png" width="70%" />
<br>
(a) ∂G/∂x of the Gaussian kernel<br>
(b) ∂G/∂x of the Gaussian kernel at y = 0<br>
<br>
The proposed loss function suppresses the influence of extreme outliers by relying on Gaussian-smoothed boundary responses rather than raw Euclidean distances.<br>
As a result, isolated noisy pixels or local GT errors do not dominate the loss value, leading to more stable optimization.


## Comparision with Hausdorff Distance Loss

<img src="assets/7.png" width="70%" />
<br>
Although both methods measure boundary discrepancies, Hausdorff distance relies on brute-force nearest-neighbor search between boundary pixels, resulting in high computational cost and sensitivity to outliers.<br>
In contrast, the proposed method achieves similar boundary sensitivity with significantly lower computational complexity.


## Quantitative Evaluation

<img src="assets/8-1.png" width="70%" />
<img src="assets/8-2.png" width="70%" />
<br>
Quantitative results show that the proposed loss function consistently outperforms conventional region-based and boundary-based loss functions across multiple evaluation metrics.<br>
The improvements are especially noticeable in boundary-sensitive metrics under noisy document conditions.

## Qualitative Evaluation

<img src="assets/9-2.png" width="100%" />
<br>
Qualitative comparisons demonstrate that the proposed method produces smoother and more coherent boundaries in challenging cases such as ink bleeding, broken strokes, and background noise.<br>
The predicted masks better preserve the overall character structure compared to baseline losses.


## Generalization

<img src="assets/10.png" width="70%" />
<br>
The model trained with the proposed loss function generalizes well to other historical documents that were not included in the training and validation set.<br>
According to the result, the model produces stable and visually consistent segmentation results, indicating practical applicability to real-world restoration scenarios.





