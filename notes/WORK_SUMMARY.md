# Work summary: multiscale hydrodynamic entropy in neural networks

This folder is a research project on whether trained neural networks organize energy across scales the way turbulent fluids do. The method follows M. K. Verma’s hydrodynamic entropy: treat a layer or mesh level as a “mode,” turn its energy into a probability, and measure how evenly that energy is spread.

There is no git history in this folder. The record of the work is the notebooks, `1. Report.docx`, `2. Presentation.pptx`, and the two FourCastNet PDFs. Numbers below are taken from saved notebook outputs, not re-run here.

## Shared method

```
For a set of scales labeled by index k:
```

1. Measure an energy E(k) at that scale (weight energy, activation energy, or a geometric projection).
2. Normalize p_k = E(k) / \sum E.
3. Compute hydrodynamic entropy S_H = -\sum p_k \log_2 p_k, and compare it with the maximum S_{\max} = \log_2 N.
4. Fit a power law E(k) \propto k^\alpha and compare \alpha with Kolmogorov -5/3 \approx -1.667 and, for the atmosphere, the Nastrom–Gage slopes (-3 at synoptic scales, -5/3 at mesoscales).

What “scale” means depends on the model:

- **CNNs:** receptive-field size R in pixels. Wavenumber is taken as k = 1/R.
- **GraphCast:** icosahedral mesh level M0–M5 or M6. Edge length sets the physical scale (about 10,000 km down to 156 km).
- **FourCastNet:** Fourier wavenumber inside the Adaptive Fourier Neural Operator (AFNO).

Two energy definitions are used, and they do not always agree:

- **Mean energy per unit** (per edge, or mean W^2) removes the bias from “there are more edges / more weights at fine scales.”
- **Total energy** (\sum W^2 or \sum h^2) grows with the number of units and can look like a steep positive cascade even when the per-unit energy is flat.



## 1. Vision networks as a control (CNNs and an MLP)

These notebooks test the entropy method on familiar image models before applying it to weather models.

### Receptive fields and mean weight energy

`Main codes/3. allcnn_mean.ipynb` and the tables in `1. Report.docx` compute receptive fields layer by layer, then the mean squared weight \mathrm{mean}(W^2) of ImageNet-pretrained convolutions.


| Model   | Conv layers | Final RF | S_H   | S_H/S_{\max} | Fit of \mathrm{mean}(W^2) vs 1/R |
| ------- | ----------- | -------- | ----- | ------------ | -------------------------------- |
| AlexNet | 5           | 163 px   | 1.323 | 0.570        | (1/R)^{1.112}                    |
| VGG-16  | 13          | 196 px   | 1.067 | 0.288        | (1/R)^{1.052}                    |
| VGG-19  | 16          | 252 px   | 1.103 | 0.276        | (1/R)^{1.010}                    |


Energy is concentrated in the first convolution. Deeper VGG nets are less uniform (S_H/S_{\max} near 0.28) than AlexNet (0.57). The reported exponents are positive because the fit is E \sim k^{+\beta} with k = 1/R: mean weight energy falls as the receptive field grows. That is the opposite sign convention from the later GraphCast fits, which report \alpha in E \propto k^\alpha and compare it directly with -5/3.

### Total weight energy on pretrained VGG

`Other codes/vgg16pretrained.ipynb` and `Other codes/vgg19_weight_analysis.ipynb` use total \sum W^2 instead of the mean. That quantity *grows* toward deeper layers, so the fitted exponent is negative and shallow:


| Model  | \alpha | R^2   | S_H/S_{\max} |
| ------ | ------ | ----- | ------------ |
| VGG-16 | −0.507 | 0.956 | 0.944        |
| VGG-19 | −0.415 | 0.947 | 0.962        |


VGG-19’s bootstrap 95% interval on \alpha is [−0.471, −0.348] (10,000 resamples, 16 layers). Local slopes vary a lot (coefficient of variation about 2), so a single power law is only a rough description. Normalized entropy near 0.96 means the total weight energy is spread almost evenly across layers. Both slopes are much shallower than Kolmogorov -1.67.

### Training from scratch

`Other codes/vgg_sample_not_pretrained.ipynb` trains a VGG-style net and snapshots \sum w^2. The power-law exponent moves from \alpha = -0.670 at initialization (R^2 = 0.943) to about -0.48 by epoch 30, still far from -5/3. Per-layer Shannon entropy of the weights drops over the first few epochs and then levels off.

`Other codes/mlp_training_dynamics.ipynb` trains a 5-hidden-layer MLP (512 units) on CIFAR-10 and tracks both activations and weights.

- In the first run, the activation exponent moves from about -3.02 at initialization to about -0.97 after 30 epochs. The notebook reads this as SGD flattening a steep random spectrum.
- A second run with bootstrap intervals (epoch 0 to 50) finds activation \alpha from -1.67 [−1.96, −1.24] to -0.93 [−1.38, −0.42], while the weight exponent steepens from -0.60 to -1.22. Weight entropy falls from S_H/S_{\max} = 0.91 to about 0.67.
- With only five layers, these fits are descriptive. Clauset-style tests are not applicable.



### ResNet-50

`Other codes/resnet50_activation_analysis.ipynb` separates activations from weights on a pretrained ResNet-50 (five stages).

- **Activations:** \alpha = -0.061, R^2 = 0.004. There is no power-law trend. The notebook attributes this to residual connections, which keep h_{l+1}^2 from collapsing.
- **Weights:** \alpha = -0.765, R^2 = 0.957, S_H = 1.715 bits, S_H/S_{\max} = 0.739. Weight energy still declines toward finer receptive fields, but less steeply than Kolmogorov.

The notebook’s comparison line is: VGG (no skip) dissipates activation energy, ResNet holds it near flat.

## 2. GraphCast

GraphCast (DeepMind) is a graph neural network on a hierarchy of icosahedral meshes. Three public checkpoints are analyzed:


| Checkpoint  | Resolution | Pressure levels | Mesh  | Edges   | Training data       |
| ----------- | ---------- | --------------- | ----- | ------- | ------------------- |
| Small       | 1.0°       | 13              | M0–M5 | 81,900  | ERA5 1979–2015      |
| Operational | 0.25°      | 13              | M0–M6 | 327,660 | ERA5-HRES 1979–2021 |
| Full        | 0.25°      | 37              | M0–M6 | 327,660 | ERA5 1979–2017      |


Latent size is 512 and there are 16 message-passing steps in each checkpoint. Mesh lengths run from about 10,008 km (M0, 60 edges) to 313 km (M5) or 156 km (M6, 245,760 edges).

### Geometric encoder spectrum (no weather input)

`Main codes/5. Graphcast-small.ipynb`, `6. Graphcast-operational.ipynb`, and `4. Graphcast_full.ipynb` do not run a forecast. They take the first linear map of the mesh-edge encoder,

`mesh_gnn/~_networks_builder/encoder_edges_mesh_mlp/~/linear_0`,

which is a single shared matrix W of shape (4, 512). Each edge has a 4-vector x = (\Delta x, \Delta x). The energy at a mesh level is the mean over edges of W x^2.


| Model       | S_H (bits) | S_{\max} | S_H/S_{\max} | \alpha | R^2    | Dominant level |
| ----------- | ---------- | -------- | ------------ | ------ | ------ | -------------- |
| Small       | 1.205      | 2.585    | 46.6%        | −1.866 | 0.9995 | M0, 70.5%      |
| Operational | 1.206      | 2.807    | 42.9%        | −1.846 | 0.9986 | M0, 70.5%      |
| Full        | 1.206      | 2.807    | 43.0%        | −1.843 | 0.9986 | M0, 70.5%      |


The three checkpoints give almost the same spectrum. About 70% of this geometric energy sits on the coarsest mesh. The slope is close to, and slightly steeper than, Kolmogorov -5/3. Because W is shared and x is purely geometric, this slope is a property of the encoder acting on edge geometry, not of a weather state.

`

1. Report.docx` has the CNN receptive-field tables and then only the headings “Graphcast / GC_Full / GC_Small / GC_Operational.” The quantitative GraphCast results live in the notebooks, not in that document.



### Activation spectrum on a real forward pass

`Other codes/graphcast-analysis.ipynb` (104 cells) loads GraphCast Small, runs a forward pass on ERA5 sample data, and hooks the processor so it can record edge features after message passing. This is the physically relevant spectrum, and it is much shallower than the geometric W\cdot x result.

Energy per edge is split as E = S + V:

- S: energy of the mean activation (the systematic signal).
- V: variance around that mean (treated as informational content).

Level-aggregated fit (6 mesh levels):


| Component  | \alpha (OLS) | Bootstrap 95% CI | R^2  |
| ---------- | ------------ | ---------------- | ---- |
| Total E    | −0.278       | [−0.564, −0.033] | 0.70 |
| Signal S   | −0.597       | [−1.152, −0.115] | 0.70 |
| Variance V | +0.155       | [+0.038, +0.261] | 0.79 |


Finer edge-length binning moves the total-energy exponent only slightly, to about -0.34 (16 bins) or -0.32 (11 bins). An earlier notebook (`finalgc_mar.ipynb`, cited but not in this folder) reported \alpha \approx +1.72 for the *sum* of h^2. That positive slope is an edge-count artifact: M5 has 1,024 times as many edges as M0. The mean per edge is the quantity used here.

Other checks in the same notebook:

- **The slope is learned.** Shuffling the mesh gives \alpha \approx 0.02. Random features give \alpha \approx 0. The negative slope is not an automatic consequence of the mesh.
- **Broken power law.** A spectral break near 544 km is preferred over a single power law (\DeltaAIC large). Large-scale slope \approx -0.04; small-scale slope \approx -0.85. Local slopes of E have coefficient of variation about 1.2, so a pure power law is a poor description.
- **Output fields, not internal edges.** 2D spectra of the forecast itself are steep: 2 m temperature \alpha = -3.51 (R^2 = 0.95), mean sea-level pressure \alpha = -4.17 (R^2 = 0.94), 10 m zonal wind \alpha = -3.11 (R^2 = 0.98). Those are closer to a synoptic -3 spectrum than the internal edge features are.
- **Distribution vs spectrum.** A Clauset–Shalizi–Newman test on the *histogram of edge energies* prefers lognormal, exponential, and stretched-exponential alternatives over a power-law distribution. That is a different question from “does mean energy versus scale follow a power law.”
- **Per message-passing step.** Stacking 6 levels × 34 recorded steps gives 204 points. The pooled exponent is \alpha = -0.061 with 95% CI [−0.110, −0.009]. The interval excludes 0, so a weak decay is detectable, and it excludes -1.67. The fit itself is weak (R^2 = 0.025). The exponent drifts across GNN steps (about -0.105 to 0).
- **Cascade direction.** Net energy flux is forward (toward small scales). Variance flux is inverse. Normalized spectral entropy of the variance is about 0.99, which the notebook reads as information kept across scales while the mean signal collapses.



## 3. FourCastNet

`Main codes/7. Fourcastnet_manual.ipynb`, `FourCastNet Spectral Analysis Report.pdf`, and `Fourcastnet_ppt.pdf` set up the same entropy analysis for NVIDIA’s FourCastNet (AFNO, 12 blocks, 0.25° grid, 20 input channels, hidden width 1024).

The pipeline:

1. Clone the official FourCastNet repository and load `AFNONet`.
2. Register forward hooks on all 12 spectral blocks under `torch.no_grad()`.
3. Map each complex activation to a radial wavenumber k = \sqrt{k_x^2 + k_y^2}.
4. Collapse |a(k)|^2 into 1D shell energy with a vectorized `scatter_add_` (or a CPU equivalent in the notebook run).
5. Compute S_H per block and a spectral slope \beta.

The notebook run that is saved used **mock ERA5** (synthetic 1/k^2 or random fields) and, in the executed path, did not yet show a full pretrained-weight trajectory. The measured shell slope was \beta \approx 0.976. For spatially white noise, energy per radial ring grows like the ring circumference, so E(k) \propto k^{+1}. Entropy across the 12 blocks moved only in the fourth decimal place (variance under 0.001 bits). That is a pipeline check: the binning is geometrically correct, and the network does not invent large-scale structure in pure noise.

The report states the hypotheses still to test on real ERA5 and pretrained `backbone.ckpt` weights:

- **Inside a block (“latent whitening”).** Physical enstrophy decays near k^{-3}. The AFNO may boost high frequencies so hidden channels do not lose variance, producing a much shallower slope.
- **Across blocks.** Soft-thresholding is expected to act like a viscosity: S_H should fall from block 0 to block 11 as the forecast becomes more organized.

Static AFNO weights are not the right object. They are shared across all wavenumbers, so |W|^2 is flat. The quantity of interest is the activation a(k).

## What the results say together

1. **The same entropy definition behaves differently in vision nets and in weather GNNs.** Mean convolutional weight energy is concentrated in early layers (especially VGG). Residual networks keep activation energy nearly flat across stages. An MLP’s activation spectrum flattens during SGD, from a steep random initialization toward a slope near -1.
2. **GraphCast’s geometric encoder looks Kolmogorov-like** (\alpha \approx -1.85, R^2 > 0.998) and is almost identical for Small, Operational, and Full. Most of that energy is on the coarsest mesh.
3. **GraphCast’s learned edge activations do not.** Mean energy per edge declines only as about k^{-0.28}. The decline is mostly in the mean signal; the variance is flat or slightly rising. Shuffled and random baselines lose the slope, so it is learned, but it is not classical turbulence and it is not a clean single power law.
4. **GraphCast’s output fields are a different story from its internal edges.** Forecast spectra of temperature, pressure, and wind are steep (\alpha about -3 to -4).
5. **FourCastNet’s measurement pipeline is validated on noise and not yet closed on real weather.** The white-noise control returns the expected k^{+1} shell spectrum and a flat entropy trajectory. The comparison to a k^{-3} enstrophy cascade on real ERA5 is written up as the next step.



## File map


| Path                                             | Role                                                                          |
| ------------------------------------------------ | ----------------------------------------------------------------------------- |
| `Main codes/3. allcnn_mean.ipynb`                | AlexNet, VGG-16, VGG-19 receptive fields and \mathrm{mean}(W^2) entropy       |
| `Main codes/4. Graphcast_full.ipynb`             | GraphCast Full geometric W\cdot x spectrum                                    |
| `Main codes/5. Graphcast-small.ipynb`            | GraphCast Small geometric W\cdot x spectrum                                   |
| `Main codes/6. Graphcast-operational.ipynb`      | GraphCast Operational geometric W\cdot x spectrum                             |
| `Main codes/7. Fourcastnet_manual.ipynb`         | FourCastNet hook, shell energy, white-noise control                           |
| `Other codes/graphcast-analysis.ipynb`           | GraphCast Small forward-pass spectrum, statistics, baselines                  |
| `Other codes/mlp_training_dynamics.ipynb`        | MLP spectrum during training                                                  |
| `Other codes/resnet50_activation_analysis.ipynb` | ResNet-50 activation vs weight energy                                         |
| `Other codes/vgg16pretrained.ipynb`              | Pretrained VGG-16 total weight energy                                         |
| `Other codes/vgg19_weight_analysis.ipynb`        | Pretrained VGG-19 weight energy and bootstrap tests                           |
| `Other codes/vgg_sample_not_pretrained.ipynb`    | VGG trained from scratch; \alpha vs epoch                                     |
| `1. Report.docx`                                 | Receptive-field tables for AlexNet, VGG-16, VGG-19; GraphCast section started |
| `2. Presentation.pptx`                           | Slide deck (not re-extracted here)                                            |
| `FourCastNet Spectral Analysis Report.pdf`       | FourCastNet method, hypotheses, and control-test status                       |
| `Fourcastnet_ppt.pdf`                            | Matching 9-slide talk, 25 April 2026                                          |


