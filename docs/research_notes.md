# 📓 Robotics Research Notes

A deep-dive log of technical insights, control theory trade-offs, and architectural decisions made during the development of this playground.

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