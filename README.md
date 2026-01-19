# Efficient Quantum State Tomography via Deep Active Learning for Continuous Variable Systems

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: In Development](https://img.shields.io/badge/Status-In%20Development-red)]()

> 🚧 **Project Status: Under Construction**
>
> This repository is currently under active development. The codebase, API interfaces, and documentation are subject to frequent updates.
> Please feel free to open an issue if you encounter bugs or have feature requests.

---

## 📝 Abstract

This project introduces a **Deep Active Learning** framework designed to revolutionize Quantum State Tomography (QST) for Continuous Variable (CV) systems.

By deploying a neural network-based intelligent agent, we replace traditional, time-consuming measurement strategies (like raster scanning) with an adaptive approach. The agent learns to navigate the phase space, identifying optimal measurement points that maximize **information gain**. This allows for the high-fidelity reconstruction of density matrices ($\rho$) and Wigner functions using a significantly reduced number of samples, thereby mitigating decoherence risks during experimentation.

## 🚀 Background & Motivation

### The Pain Point
Traditional quantum state tomography, particularly **Wigner Tomography**, faces significant challenges when scaling to high-dimensional Hilbert spaces (such as Bosonic modes used for GKP states):

* **Inefficient Sampling:** Traditional methods rely on **Raster Scans** or random sampling, which suffer from high sample complexity.
* **Time Constraints:** Acquiring a sufficient number of data points for high-fidelity reconstruction is experimentally expensive.
* **Decoherence:** The long duration required for comprehensive measurement often exceeds the coherence time of the quantum state, degrading the quality of the result.

### The Solution
We propose a shift from static sampling to **Active Sensing**. Instead of blindly scanning the phase space, our Neural Network Agent dynamically decides "where to measure next."

* **Smart Sampling:** The agent learns a policy to prioritize measurements that reduce state uncertainty the most.
* **Sparse Reconstruction:** Achieves high-fidelity state reconstruction from sparse data.
* **Accelerated Workflow:** Drastically reduces total experiment time, preserving the integrity of fragile quantum states.

## ✨ Key Features

* **Deep Active Learning Agent:** A reinforcement learning/active learning model trained to optimize measurement trajectories.
* **CV System Focus:** Specifically optimized for Bosonic modes and high-dimensional states like **GKP (Gottesman-Kitaev-Preskill) codes**.
* **High Fidelity:** Reconstructs density matrices $\rho$ with high accuracy despite sparse input data.
* **Phase Space Optimization:** Intelligent navigation of the Wigner function landscape.

## 🛠️ Methodology

1.  **Simulation Environment:** The agent interacts with a simulated quantum environment (e.g., based on Master Equation solvers).
2.  **Policy Learning:** The neural network learns a policy $\pi(s)$ mapping the current belief state to the next optimal displacement/rotation for measurement.
3.  **Reconstruction:** A generator network or convex optimization solver reconstructs the state $\rho$ based on the selected measurements.

## 📦 Installation

```bash
# Clone the repository
git clone [https://github.com/yourusername/deep-active-qst.git](https://github.com/yourusername/deep-active-qst.git)

# Navigate to the directory
cd deep-active-qst

# Install dependencies
pip install -r requirements.txt
```
## 💻 Usage
* **Note:** Detailed documentation on hyperparameters and training configurations can be found in the docs/ directory.

1.  **Training the Agent**
Train the active learning model for a specific target state (e.g., Cat State or GKP State):

```bash
# Example: Train for GKP state with a Hilbert space dimension of 100
python train_agent.py --state_type GKP --dim 100 --method active_learning
```

2.  **Running Tomography Simulation**
Load a pretrained model and execute adaptive tomography:
print(f"Fidelity: {result.fidelity}")

from active_qst import Agent, Simulator

```bash
# Initialize environment (Continuous Variable mode)
env = Simulator(mode='continuous_variable')

# Load the pretrained Agent
agent = Agent.load('pretrained_models/gkp_agent.pth')

# Run adaptive tomography with a limit on measurement steps
result = agent.reconstruct(env, max_measurements=500)

# Output the results
print(f"Reconstruction Fidelity: {result.fidelity:.4f}")
print(f"Number of Samples Used: {result.num_samples}")
```

## 📚 Citation
If you use this code in your research, please cite:

```bash
@misc{yourname2026efficient,
  title={Efficient Quantum State Tomography via Deep Active Learning for Continuous Variable Systems},
  author={Your Name and Collaborators},
  year={2026},
  publisher={GitHub},
  journal={GitHub repository},
  howpublished={\url{[https://github.com/yourusername/deep-active-qst](https://github.com/yourusername/deep-active-qst)}}
}
```

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
