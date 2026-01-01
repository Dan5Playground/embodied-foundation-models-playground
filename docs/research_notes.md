# 📓 Robotics Research Notes

A deep-dive log of technical insights, control theory trade-offs, and architectural decisions made during the development of this playground.

---
## 🧠 Policy Architecture & Baseline Training

**Date:** December 31, 2025

**Topic:** Implementation of a Vision-to-Action Baseline Policy  
  
---

### 🏗️ Technical Architecture
For the initial baseline, I developed a **Behavioral Cloning (BC)** agent that utilizes a standard regression head to map visual observations to predicted motor commands.

#### 1. Vision Encoder (The "Eyes")
* **Backbone:** ResNet-18 (Pre-trained on ImageNet).
* **Function:** Squeezes a $96 \times 96$ RGB image into a latent feature vector of size 512.
* **Spatial Reduction:** Utilized `Global Average Pooling` followed by a `.flatten(1)` operation to transform 4D convolutional feature maps into 1D semantic vectors.

#### 2. Policy Head (The "Brain")
* **Model Type:** Multi-Layer Perceptron (MLP).
* **Action Chunking:** Instead of predicting a single next action, the model outputs a sequence of $k=16$ future actions.
* **Output Dimension:** 32 ($16 \text{ timesteps} \times 2 \text{ [x, y] coordinates}$).
* **Reshaping:** Utilized `.view(-1, 16, 2)` to reconstruct the flat MLP output into a structured temporal trajectory.

[Placeholder: Image of a Behavioral Cloning neural network architecture showing image input, a CNN encoder, and an MLP policy outputting action sequences]

---

### 🔬 Optimization & Control Theory

#### Z-Score Normalization
During initial training, the MSE loss was observed at **~24,000.0**. This was identified as a scale mismatch between the MLP's weight initialization and the raw pixel coordinates of the Push-T environment ($0-512$).
* **Solution:** Implemented normalization within the `DataLoader` using dataset-wide statistics: $a_{norm} = \frac{a - \mu}{\sigma}$.
* **Result:** Initial loss stabilized to **~0.25**, providing a smoother gradient surface for the Adam optimizer.

#### Receding Horizon Control (RHC)
The current rollout script executes the first action of the predicted 16-step chunk and re-observes the environment at the next timestep. 
* **Pros:** High reactivity to environmental changes (e.g., the block slipping).
* **Cons:** Potential for high-frequency "jitter" as the model slightly changes its plan at every frame.

---

### 🚀 State-of-the-Art (SOTA) Landscape
In professional robotics research, this MLP-Baseline is the starting point. Below are the SOTA alternatives for higher performance:

#### SOTA Vision Encoders
| Encoder | Advantage | Context |
| :--- | :--- | :--- |
| **DINOv2** | Self-supervised; excellent spatial awareness without labels. | General manipulation. |
| **ViT (Vision Transformer)** | Handles high-resolution global context via self-attention. | Used in **RT-X / OpenVLA**. |
| **R3M** | Pre-trained specifically on human ego-centric video data. | Cross-embodiment learning. |

#### SOTA Policy Heads
| Policy | Methodology | Why it Wins |
| :--- | :--- | :--- |
| **Diffusion Policy** | Denoising Diffusion Probabilistic Models (DDPM). | Handles **multimodality** (e.g., multiple ways to push a block). |
| **ACT (Action Chunking Transformer)** | CVAE + Transformer Encoder/Decoder. | Smoother trajectories and better long-term planning. |
| **Flow Matching** | Continuous-time generative modeling. | Faster inference than Diffusion with similar expressive power. |

[Placeholder Image of Diffusion Policy for robotics showing the iterative denoising process from random noise to a smooth action trajectory]


---

## 🛠️ Data Engineering: Action Chunking
**Date:** Dec 30, 2025  
**Topic:** Mitigating Compounding Errors in Imitation Learning

In standard Behavioral Cloning (BC), predicting a single action $a_t$ from observation $o_t$ leads to "drifting" where small errors accumulate, moving the robot into states not covered by the training distribution.

### Implementation: The Sliding Window
To solve this, I implemented a chunking utility (`src/utils/chunking.py`) that transforms expert trajectories into overlapping windows of size $k=16$. 
* **Input:** `(N, action_dim)`
* **Output:** `(N - k + 1, k, action_dim)`



---

## 🏎️ Control Theory: Temporal Aggregation
**Topic:** Smoothing and Latency in Generative Policies

When performing inference, the model predicts a full chunk of 16 actions at every timestep. This results in multiple overlapping predictions for any given time $t$.

### The Latency-Accuracy Trade-off
There is a fundamental tension in how we aggregate these overlapping chunks:
1. **Freshness:** Newer chunks are informed by the most recent camera observations.
2. **Consistency:** Older chunks represent a plan the robot is already executing.

### Decision: Exponential Weighting
I opted for an exponential moving average (EMA) approach, weighting the **most recent** predictions more heavily:
$$w_i = e^{-m \cdot i}$$
* **Why?** Even though newer predictions might have "drifted," they are the only ones that have seen the latest physical changes in the environment (e.g., the block slipping). The 1st action of a new chunk is statistically more accurate than the 16th action of an old chunk.

---

## 🔧 Environment Troubleshooting (M3 Apple Silicon)
**Topic:** Library Synchronization (TorchCodec + FFmpeg)

Significant effort was spent resolving the `@rpath/libavutil` link errors on macOS. 
* **Insight:** `torchcodec` is highly version-sensitive. Standardizing on **PyTorch 2.5.1** and **FFmpeg 6.1** provided the most stable "Goldilocks" environment for robotics work on M3 hardware.
* **Key Fix:** Using `conda-forge` for FFmpeg to ensure `.dylib` files are contained within the environment prefix.
* **Precision Casting:** Python/NumPy defaults to `float64`, which is incompatible with many MPS-accelerated operations. Forced explicit casting to `float32` in the `Dataset` class.