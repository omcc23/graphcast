# Project summary

This project asks whether a single number from fluid physics, hydrodynamic entropy, says something useful about neural networks.

The number measures how evenly energy is spread. It is high when energy is shared across many scales, and low when a few scales hold almost all of it. The same formula has been applied in two ways.

- On weather and vision models, the scales are layers, mesh levels, or frequency rings. The energy is the size of the weights or of the activations at that scale.
- On a small image network trained from scratch, the scales are the spatial frequencies of a hidden map. This is the experiment with the clearest result.

The write-up of that second experiment, with the graphs, is `notes/entropy_of_networks.pdf`. A longer record of the earlier measurements is `notes/WORK_SUMMARY.md`.

# Major experiments

## 1. Image networks, energy across layers

Pretrained AlexNet, VGG-16, and VGG-19. Energy is the average squared weight of each convolution, and the scale is how large a region that layer can see.

A separate check uses the total squared weight instead of the average. ResNet-50 is split into activation energy and weight energy. A VGG-style net and a five-layer fully connected net are also watched while they train on CIFAR-10.

## 2. GraphCast

Three public checkpoints: Small, Operational, and Full. Two different spectra.

- Geometric encoder only. No weather is passed in. Energy is what the first linear map does to mesh-edge geometry.
- A real forward pass of GraphCast Small. Energy is the mean activation on edges after message passing, split into the mean signal and the variation around it. Shuffled meshes and random features are the controls. Forecast fields (temperature, pressure, wind) are measured separately from the internal edges.

## 3. FourCastNet

Twelve Fourier blocks. Energy is binned into frequency rings and the entropy is computed in each block. The run that exists used synthetic noise, not a real weather state. It checks that the binning is right.

## 4. Hidden maps while a network trains

A small convolutional network, about 1.15 million parameters, on CIFAR-10. Three hidden maps, at 32, 16, and 8 pixels. Entropy is measured on those maps after every epoch. It is not part of the training loss.

The comparison is real labels against one fixed shuffle of the labels. Cropping, flipping, and a penalty on large weights are turned on or off so those choices are not confused with the labels. Extra runs use the full training set, three other starting-weight schemes, and training sets with 20% or 40% of the labels flipped.

Nineteen runs finished on two T4 GPUs in about 1.1 hours. Histories and checkpoints are in `kaggle/session-output`.

# Main results

## Hidden maps

On the last map, entropy starts near 0.83 in every run. The scale runs from 0 to 1.

| Setting | Train accuracy | Test accuracy | Entropy at the end |
| --- | ---: | ---: | ---: |
| Real labels, with augmentation | 0.95 | 0.81 | 0.76 |
| Real labels, no augmentation | 1.00 | 0.77 | 0.79 |
| Shuffled labels, with augmentation | 0.14 | 0.10 | 0.79 |
| Shuffled labels, no augmentation | 0.48 | 0.10 | 0.90 |
| Full set, real labels | 0.98 | 0.90 | 0.76 |
| Full set, shuffled labels | 0.16 | 0.09 | 0.88 |
| 20% of labels flipped | 0.69 | 0.74 | 0.77 |
| 40% of labels flipped | 0.49 | 0.67 | 0.77 |

What this says:

- With real labels, the last map becomes more concentrated and stays that way. Test accuracy climbs to 0.81 on the subset and 0.90 on the full set.
- With a pure shuffle, and with augmentation turned off, the last map becomes more spread out and finishes at 0.90. Test accuracy stays at chance, about 0.10.
- That rise is already visible while accuracy on the shuffled labels is still only about 0.15.
- Shuffling alone does not cause it. Turning augmentation off alone does not cause it. Real labels with no augmentation still end at 0.79, even when the training set is fit perfectly.
- The first map moves the other way under real labels: its entropy rises, from about 0.70 to about 0.88. The middle map moves much less.
- Other starting weights, and 20% or 40% wrong labels, stay with the real-label result.

## Earlier measurements

| What was measured | Result |
| --- | --- |
| AlexNet, average weight energy across layers | Entropy is 0.57 of its maximum |
| VGG-16 and VGG-19, average weight energy | Entropy is about 0.28 of its maximum. The first layer holds most of it |
| VGG, total weight energy instead of the average | The ratio jumps to about 0.95, because later layers have more weights |
| ResNet-50 activations | Almost flat across stages. Residual links keep the energy from collapsing |
| ResNet-50 weights | Energy still declines toward finer scales |
| Fully connected net during training | Activation energies spread out. Weight energies become less even |
| GraphCast geometric encoder, three checkpoints | Almost the same spectrum. Entropy about 1.21 bits, 43% to 47% of the maximum. About 70% of the energy is on the coarsest mesh. The slope is close to the classical turbulence value |
| GraphCast learned edge activations | Much flatter. Mean energy falls only gently toward fine scales. The variation around the mean stays spread out. Shuffling the mesh removes the slope, so it is learned |
| GraphCast forecast fields | Steep. Temperature, pressure, and wind fall off much faster with scale than the internal edges do |
| FourCastNet on noise | Frequency-ring energy grows the way empty noise should, and entropy barely changes across the twelve blocks. A real weather input has not been measured yet |

# Where to look

| Path | What it is |
| --- | --- |
| `notes/entropy_of_networks.pdf` | Question, graphs, and result for the hidden-map experiment |
| `notes/WORK_SUMMARY.md` | Record of the earlier vision, GraphCast, and FourCastNet measurements |
| `experiments/entropy.py` | The entropy calculation |
| `kaggle/session-output` | Histories, plots, and checkpoints from the nineteen training runs |
