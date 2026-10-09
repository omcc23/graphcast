# Three GraphCast measurements

These sit beside the notebooks already in `Main codes` and `Other codes`. Each folder has the note, the code, and the saved numbers. The checkpoints are not included. The scripts download them.

`1_encoder_weights` is the mesh-edge encoder on the real lengths M0–M5 or M0–M6. No weather field is loaded. The energy of a level is the mean squared output of the shared weights on the edges of that length, reported for the first linear map, both linear maps, and the output after layer norm. Entropy is in nats. The note is `approach_1.pdf`.

`2_latent_spectrum` is one GraphCast Small forecast. The latent on the mesh nodes is expanded in spherical harmonics after the encoder and after each of the 16 processor steps. The six probabilities are bands of spherical-harmonic degree lined up with the six mesh lengths. The note is `approach_2.pdf`.

`3_processor_edges` is the same forecast. The energy of a level is the mean squared output of that processor step's edge network, before the residual is added, on the edges of that length. The note is `approach_3.pdf`.
