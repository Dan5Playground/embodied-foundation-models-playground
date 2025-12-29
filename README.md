# 🤖 Embodied Foundation Models Playground

A modular research playground for exploring embodied foundation models for robotics, with a focus on end-to-end perception-to-action policies.

This repository serves as both a technical journal and a working codebase for building, running, and analyzing modern robot learning systems. 

---

## 🧠 Core Research Pillars
* **VLA (Vision-Language-Action):** Implementing multimodal transformers that ground natural language instructions into physical robot trajectories.
* **Diffusion Policies:** Leveraging generative modeling to handle multimodal expert demonstrations and ensure smooth, non-linear action sequences.
* **Imitation Learning (IL):** Scaling robotic skill acquisition through expert data and behavioral cloning rather than manual reward engineering.

---

## 📂 Repository Structure
```text
embodied-foundation-models-playground/
├── assets/             # Media (rollout videos, GIFs, architecture diagrams)
├── config/             # Hydra/YAML configuration for model hyperparameters
├── scripts/            # Entry points for experiments and training sessions
├── src/                # Modular, reusable source code
│   ├── agents/         # Architectures (MLP, Diffusion, VLA Policy heads)
│   ├── envs/           # Environment wrappers (Push-T, MuJoCo)
│   └── utils/          # Processing primitives (Action Chunking, Normalization)
└── notebooks/          # Exploratory research and math prototyping


## 📓 Technical Journal & Milestones

### 🟢 Phase 1: Environment & Perception
* **Hardware-Aware Setup:** Configured a native ARM64 environment on macOS (M3) utilizing **MPS (Metal Performance Shaders)** for GPU-accelerated inference.
* **Simulation Sandbox:** Integrated the **Push-T** benchmark for validating imitation learning algorithms.
* **Vision-Action Loop:** Verified the perception pipeline by rendering RGB observations into high-fidelity rollout videos.
* **Status:** Complete. `scripts/day1_env_test.py` successfully generates verified environment rollouts.

### 🟡 Phase 2: Action Primitives & Behavioral Cloning
* **Action Chunking:** Implementing temporal aggregation to mitigate compounding errors in long-horizon tasks.
* **Baseline IL:** Training MLP-based policies to establish performance benchmarks.
* **Status:** *In Progress*

---

## 🛠️ Installation & Setup

### 1. Prerequisites
- **Miniforge (ARM64):** Recommended for native Apple Silicon support.
- **FFmpeg:** Required for video encoding and rollout visualization.

### 2. Environment Setup
```bash
# Create the environment
conda create -n robo-vla python=3.10 -y
conda activate robo-vla

# Install dependencies
pip install -r requirements.txt

# Verification
python scripts/pushT_env_test.py