# 📓 Robotics Research Notes

A deep-dive log of technical insights, control theory trade-offs, and architectural decisions made during the development of this playground.

---
## 📝 VLA Evolution & SOTA Benchmarking

**Date:** January 3, 2026

**Topic:** Transition from a uni-modal **Diffusion Policy** to a multi-modal **Vision-Language-Action (VLA)** model. This allows the agent to condition its generative denoising process on high-level human instructions, moving the robot from "fixed behavior" to "instruction following."

---

### 2. Architecture: Triple-Conditioned 1D U-Net
We modified the 1D U-Net architecture to integrate three distinct streams of information using **FiLM (Feature-wise Linear Modulation)**.

#### Conditioning Streams:
* **Visual:** ResNet-18 backbone providing spatial context.
* **Temporal:** Sinusoidal or Linear time embeddings informing the denoising step $k$.
* **Linguistic:** CLIP (ViT-B/32) text embeddings providing semantic intent.

#### Modality Alignment:
Alignment is achieved by concatenating the 512-dim visual vector, the 128-dim time vector, and the 128-dim projected text vector into a **768-dimensional conditioning bottleneck**. This bottleneck modulates the U-Net's residual blocks, "steering" the action trajectory based on the text prompt.



---

### 3. SOTA VLA Landscape: Beyond Diffusion

While our implementation uses Diffusion, the industry is shifting toward massive foundation models that align modalities through shared transformer vocabularies or flow matching.

#### A. Physical Intelligence (π): π₀ and π₀.₅
* **Core Logic:** Uses **Flow Matching** instead of Diffusion. Flow matching learns a direct, straight-line vector field from noise to data, making inference significantly faster (often requiring only 1–3 steps).
* **Alignment:** π₀ uses a large Pre-trained Vision-Language Model (VLM) as a backbone. It treats the robot as an "output head" of a model that already understands the world through internet-scale text and images.

#### B. 1X Technologies: EVE & NEO
* **Core Logic:** Focuses on **World Models**. The model is trained to predict the next video frame and the next action simultaneously, ensuring the robot understands the physical consequences of its moves.
* **Alignment:** Utilizes a **Shared Latent Space**. Through self-supervised learning on massive teleoperation data, vision and language are mapped to a latent state that represents physical "affordances" (e.g., understanding that "pick up" requires a specific gripper state).

#### C. NVIDIA: VILA & Project GR00T
* **Core Logic:** NVIDIA’s **VILA** (Visual-Language-Action) excels at multi-modal reasoning, allowing the robot to interpret complex constraints (e.g., "The surface is hot, move the cup carefully").
* **Alignment:** **Cross-Attention Transformers**. NVIDIA treats image patches and text tokens as sequences. In models like GR00T, actions are **discretized into tokens**, effectively turning robot control into a "translation" task where the model translates pixels and English into "motor-language".



---

### 4. Comparison of Alignment Strategies

| Strategy | Mechanism | SOTA Example |
| :--- | :--- | :--- |
| **Feature Fusion** | Concatenating V + L vectors | Early Diffusion Policies |
| **Cross-Attention** | Action "queries" Vision/Language "keys" | NVIDIA VILA, Octo |
| **Action-as-Tokens** | Actions are words in a shared vocabulary | Google RT-2, NVIDIA GR00T |
| **Flow Matching** | Direct vector field from noise to action | Physical Intelligence (π₀) |

---

### 5. Summary of Alignment Logic
The SOTA "bridge" is a **Transformer Backbone**.
1.  **Vision** is tokenized into patches.
2.  **Language** is tokenized into words.
3.  **The Transformer** uses attention to learn that the word "T-block" in the text refers to a specific cluster of pixels in the image.
4.  **The Action Head** then projects this "aligned understanding" into motor coordinates ($X, Y$).

---
## 🎮 Generative Behavior Cloning via Diffusion Policy

**Date:** January 2, 2026

**Topic:** Transition from a deterministic Multi-Layer Perceptron (MLP) to a **Generative Diffusion Policy**. This move allows the robot to represent "multimodal" distributions—essentially learning that there are multiple valid paths to solve the Push-T task rather than just averaging them into a single (often failing) path.

---

### 🏗️ The 1D Temporal U-Net Architecture
The "brain" of our Diffusion Policy is a 1D U-Net designed to denoise action sequences. Unlike 2D U-Nets used in image generation, this model treats the robot's action horizon as a temporal sequence.

#### Architecture Details:
* **Backbone:** ResNet-18 (with the final FC layer removed) acts as the visual feature extractor ($1 \times 512$ embedding).
* **Action Horizon:** 16 steps (predicting $X, Y$ coordinates for the current and next 15 frames).
* **Conditioning (FiLM):** We used **Feature-wise Linear Modulation**. The visual features and the diffusion timestep are fused into a conditioning vector that scales and shifts the activations in the U-Net.
* **Layers:**
    * **Downsampling:** 1D Convolutions with kernel size 5 to capture broad temporal context.
    * **Bottleneck:** Deepest layer where visual context is most heavily integrated.
    * **Upsampling:** Reconstructs the high-resolution 16-step trajectory using skip connections to preserve fine-grained control details.
* **Activation:** `nn.Mish()` was used throughout for smoother gradients during the iterative denoising process.

---
### 📈 Training Summary
* **Loss Function:** Mean Squared Error (MSE) between the added Gaussian noise ($\epsilon$) and the predicted noise ($\epsilon_\theta$).
* **Normalization:** Critical implementation of Z-score normalization using dataset statistics. (Unnormalized actions led to initial losses > 100; normalized loss settled at ~0.09).
* **Scheduler:** DDPMScheduler using a `squaredcos_cap_v2` schedule to maintain trajectory structure during the diffusion process.
---

### 🚀 SOTA Alternatives 
If hardware constraints were removed, the following architectures represent the current frontier:

1.  **Flow Matching:** Instead of Gaussian diffusion, Flow Matching learns a straight-line vector field from noise to data. It is significantly faster, often requiring only **1–3 steps** of inference.
2.  **Diffusion Transformers (DiT):** Replacing the U-Net with a Transformer backbone. This allows the model to scale with much larger datasets and handle multi-modal inputs (e.g., 3+ camera feeds) via cross-attention more effectively.
3.  **Consistency Models:** These models are trained to map any noise level to the final "clean" action in a **single step**, enabling extremely high-frequency control (100Hz+) without the computational overhead of iterative denoising.



---
##  🕹️ Control Refinement and Quantitative Evaluation

**Date:** January 1, 2026

**Topic:** Transition from subjective visual assessment to objective performance metrics 
  
### 🛠️ Implementation: Temporal Aggregation (EMA)
A **Temporal Aggregation** wrapper was implemented using **Exponential Moving Average (EMA)** weighting. 

* **The Logic:** Instead of discarding the 16-step action chunk at every frame (Receding Horizon), the controller now maintains a rolling buffer of overlapping plans.
* **The Math:** A weighting constant $k=0.25$ was applied to blend the current prediction with historical predictions.
* **The Benefit:** This ensures that the robot’s "intent" remains consistent across timesteps, significantly smoothing the end-effector trajectory and reducing the "shivering" effect common in naive Behavioral Cloning.

---

### 📊 Results & Observations
A head-to-head comparison was conducted between the **Basic Receding Horizon** controller and the **EMA Smooth** controller over 5 episodes each.

| Metric | Basic Controller | EMA (Smooth) Controller |
| :--- | :--- | :--- |
| **Success Rate** | 0.0% | 20.0% |
| **Avg Max Reward** | 0.137 | 0.418 |
| **Movement Quality** | High-frequency jitter, unstable | Fluid, human-like, purposeful |

**Key Observations:**
1. **Control vs. Intelligence:** The 3x improvement in Average Max Reward confirms that even with an identical "brain" (weights), the control strategy is a critical factor in success.
2. **The "Contact" Problem:** Many failures (Reward: 0.0) occurred because the model failed to make initial contact with the "T" block. This suggests the MLP is highly sensitive to initial distribution shifts (the "compounding error" problem).
3. **The Multimodal Ceiling:** Despite the smoothing, the model often fails to choose a single consistent side of the block to push. Because the expert data contains multiple ways to solve the task, the MLP head tries to "average" these paths, resulting in ineffective, "middle-of-the-road" movements.

The current MLP baseline has reached its functional limit. While EMA smoothing made the policy 20% successful, the "averaging" behavior of the MLP head is preventing it from reaching SOTA performance (80%+).

**Next Steps:**
* **Architecture Upgrade:** Transition to **Diffusion Policy** to handle multimodal action distributions (allowing the model to "commit" to one path).
* **Robustness:** Implement **Observation Augmentation** (random cropping/color jitter) to increase the model's tolerance for varied starting positions.


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