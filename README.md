# Class-Conditional Diffusion Model

## Introduction

The aim of this study is to understand diffusion models and the probabilistic reasoning behind them, as one of the core methods driving modern generative AI. A class-conditional diffusion model is implemented from scratch and trained on natural colour images of animal faces, so the model can be told which animal to draw. Implementing the model, running a series of experiments, and comparing their outcomes builds deeper knowledge of probability, inference, and the machine-learning concepts behind today's generative image models.

## Background
A diffusion model generates new images by learning to reverse a process that gradually destroys an
image with noise. It is built from two processes that run in opposite directions along the same chain of
increasingly noisy images.

**The forward process** X0 is the original data, and X1 is created by adding Gaussian noise to X0. This action is repeated t times until Xt is created. The final data Xt is complete noise — for an image, Xt is an indistinguishable blur (see t = 999 above).

The main reason for using Gaussian noise is that it is smooth and continuous, with a mean of 0 and a standard deviation of 1. It affects all features of the data only slightly, so each noisy version is only slightly different from the previous one, and the original data changes gradually into complete noise. This gradual, well-behaved corruption is what makes the denoising (reverse) action possible to learn. Here a cosine (squaredcos_cap_v2) schedule controls how fast the noise is added.

![Forward process](figs/forward_diffusion.png)

**Fig. 1.** Forward process — a real "dog" image is gradually corrupted into pure noise.

**The reverse process** Initially, the model forecasts the noise that was added during the forward process. Then it creates Xt-1 by removing the predicted noise from Xt. This denoising action is repeated until the noise is removed. The output of the reverse process is similar to the original data but is a new image, because the predicted noise differs slightly from the noise added in the forward process — so a different, plausible animal face is produced each time.

![Reverse process](figs/reverse_diffusion.png)

**Fig. 2.** Reverse process — starting from noise and conditioning on "dog", a clean face emerges.

## Method

**Data.** Training uses the AFHQ dataset of aligned, front-facing animal faces. The training split holds
**16,130 images** (5,653 cat / 5,239 dog / 5,238 wild), obtained from the HuggingFace Hub
([`huggan/AFHQ`](https://huggingface.co/datasets/huggan/AFHQ)). Images are resized to **128×128**,
randomly flipped horizontally, and normalised to the range [-1, 1].

![Real samples](figs/real_samples.png)

**Fig. 3.** Real AFHQ training samples.

**Model.** The denoising network is a **U-Net** (`UNet2DModel` from HuggingFace `diffusers`) that
predicts the noise at each step. It has six resolution levels (128 → 64 → 32 → 16 → 8 → 4) with two residual
blocks per level and self-attention at the 16×16 and 8×8 stages, giving about **129.2M parameters**.
Skip connections carry fine detail from the downsampling path to the upsampling path.

**Class control.** The class label is supplied as a learned embedding, so the network learns a separate
behaviour per class. To enable **classifier-free guidance (CFG)**, the label is randomly dropped for
about 10% of training images and replaced with a special "no-class" token. The same network therefore
learns both conditional generation ("draw a dog") and unconditional generation ("draw any animal"); at
sampling time the two are combined, and a guidance scale controls how strongly the output is pushed
toward the requested class.

| Component | Setting |
|---|---|
| Architecture | U-Net, 6 levels (128→64→32→16→8→4), attention at 16×16 and 8×8 |
| Parameters | (128, 128, 256, 384, 512, 512) channels, 2 blocks/level — ~129.2M |
| Conditioning | 3 classes + 1 "no-class" token, 10% label dropout (CFG) |
| Diffusion | 1000 steps, cosine (`squaredcos_cap_v2`) schedule, ε-prediction |
| Loss | noise-prediction MSE, Min-SNR-γ weighting (γ = 5) |
| Optimiser | AdamW, lr 2e-4, warmup + cosine decay, gradient clipping, fp16 |
| Stabilisation | EMA weights (decay 0.9995) |
| Sampling | DDIM, 100–200 steps, guidance scale ≈ 1.5 |

The full pipeline is in [`notebook/afhq_diffusion.ipynb`](notebook/afhq_diffusion.ipynb) and runs top to
bottom (`pip install -r requirements.txt`, then run all). After training, images are generated with a
single call, e.g. `draw("cat", n=8)`.


## Results

**Training.** The noise-prediction loss (Min-SNR weighted) falls sharply from **0.111** at epoch 1 to
about **0.018** by epoch 2, then decreases slowly to **0.0113** by epoch 50. The curve is smooth with no
spikes, and is essentially flat after about epoch 20 — most learning happens early.

![Loss curve](figs/loss_curve.png)
**Fig. 4.** Conditional training loss per epoch.

**Conditional generation.** The model reliably produces the requested class. In Fig. 5 each row is a
single class; the classes are clearly separable and retain variety rather than collapsing onto one
prototype face.

![Generated images](figs/class_sheet.png)

**Fig. 5.** Generated images, one row per requested class (cat / dog / wild).

**Effect of guidance.** Increasing the guidance scale trades diversity for class typicality. Low values
give softer, more varied faces; high values give crisper, more "typical" ones that eventually
over-saturate and lose variety. A value around 1.5 is a good balance.

![Guidance sweep](figs/guidance_sweep.png)

**Fig. 6.** Guidance sweep for the "dog" class (scale = 1.0 / 1.5 / 3.0).

Additional experiments in the notebook show that a cosine schedule outperforms a linear one at this
resolution, a larger model produces clearer samples than a narrower one on the same budget, and
interpolating between two class embeddings yields plausible in-between animals.
<br><br>

## Discussion

The results show that a single conditional network with classifier-free guidance is enough for reliable,
controllable class generation, and that standard techniques (cosine schedule, Min-SNR weighting, EMA,
gradient clipping) give smooth, stable training. At 128×128 the samples are sharp and clearly
class-separable — a clear step up from a 64×64 baseline. Two limitations remain. First, fine details
(fur, whiskers, eye highlights) are good but not photoreal; pushing further would mean a higher
resolution still, more data, or longer training. Second, occasional unnatural background tints come from
the variety of AFHQ's source photos, not from undertraining. A quantitative metric such as FID would
complement the qualitative results presented here.


## Future Research

A future extension of this project would be to move from class-conditional generation to attribute- or
text-conditional generation. The current model can control broad animal classes such as cat, dog, and
wild, but it cannot understand detailed prompts such as "a dog wearing a red hat" because it is trained
only with class labels. To support this kind of control, the model would need either additional labelled
attributes, such as colour and accessories, or a text-conditioning mechanism similar to modern
text-to-image diffusion models. This would allow more flexible and detailed image generation beyond
simple class labels.


## Conclusion

A class-conditional diffusion model was implemented and trained from scratch on AFHQ, generating clear,
class-separable cat, dog, and wild faces on demand through classifier-free guidance. The experiments
clarified how guidance strength, noise schedule, model size, and training length affect image quality,
and showed that most learning occurs early in training. Beyond the working model, the project gave a
concrete understanding of the probabilistic ideas behind diffusion — a fixed noising process inverted by
a learned noise predictor — producing sharp, controllable 128×128 results.


## Reference

- J. Ho, A. Jain, and P. Abbeel, "Denoising Diffusion Probabilistic Models," *NeurIPS*, 2020.
  https://arxiv.org/abs/2006.11239
- J. Ho and T. Salimans, "Classifier-Free Diffusion Guidance," 2022. https://arxiv.org/abs/2207.12598
- T. Hang et al., "Efficient Diffusion Training via Min-SNR Weighting Strategy," *ICCV*, 2023.
  https://arxiv.org/abs/2303.09556
- Y. Choi, Y. Uh, J. Yoo, and J.-W. Ha, "StarGAN v2: Diverse Image Synthesis for Multiple Domains"
  (AFHQ dataset), *CVPR*, 2020. https://github.com/clovaai/stargan-v2
- HuggingFace `diffusers` library. https://github.com/huggingface/diffusers
- C. University, "Diffusion and Flow Matching Models," *Lecture 6*, Senjian, Ed.
- M. Wornow, "Diffusion Models from Scratch," michaelwornow.net, 2023.
  https://michaelwornow.net/2023/07/01/diffusion-models-from-scratch
