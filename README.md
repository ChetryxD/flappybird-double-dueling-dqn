# 🐦 Flappy Bird AI — Double Dueling DQN (PyTorch)

> A from-scratch implementation of a **Double Dueling Deep Q-Network (DQN)** trained to master Flappy Bird using value-based reinforcement learning.

![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
![RL](https://img.shields.io/badge/Reinforcement-Learning-green)

---

## 🎮 Demo

![flappybirds](https://github.com/user-attachments/assets/658bd3f9-401e-4e47-a63d-d14c9ca817a5)


---

## 📌 Overview

This project implements a **Double Dueling Deep Q-Network (DQN)** to solve Flappy Bird in a sparse-reward environment.

To achieve stable long-horizon learning, the agent integrates:

- **Experience Replay**
- **Target Network Synchronization**
- **Double DQN** (reduces Q-value overestimation)
- **Dueling Architecture** (separates state-value and advantage streams)
- **Huber Loss (SmoothL1Loss)** for stable updates

The result is a stable agent capable of consistently passing dozens of pipes.

---

## 🚀 Performance

| Metric | Value |
|--------|--------|
| Peak Score | **80–120+ pipes** |
| Consistent Average | **40–70 pipes** |
| Convergence | ~300k episodes |
| Training Time | ~3–4 hours (CPU) |

✅ Stable learning curve  
✅ Reduced Q-value oscillations  
✅ Strong generalization beyond early episodes  

---

## 📈 Training Curve

<img width="788" height="587" alt="image" src="https://github.com/user-attachments/assets/f4a73851-ca98-4ed6-a24f-1e831927a083" />

---

## 🧠 Algorithmic Architecture

| Technique | Purpose |
|------------|----------|
| Experience Replay | Breaks temporal correlation between samples |
| Target Network | Stabilizes Q-learning updates |
| Double DQN | Mitigates overestimation bias |
| Dueling DQN | Separates state-value from action advantage |
| Huber Loss | Prevents gradient instability |

### Dueling Decomposition

The Q-value is computed as:

```python
Q(s, a) = V(s) + (A(s, a) - mean(A(s, ·)))
📂 Project Structure

dqn_pytorch/
│
├── src/
│   ├── agent.py
│   ├── dqn.py
│   └── experience_replay.py
│
├── configs/
│   └── hyperparameters.yml
│
├── assets/
│   ├── learning_curve.png
│   └── flappybird_demo.gif
│
├── runs/               # training outputs (gitignored)
│
├── requirements.txt
└── README.md

⚙️ Installation

git clone https://github.com/ChetryxD/flappybird-double-dueling-dqn.git
cd flappybird-double-dueling-dqn

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt

🏋️ Training

python -m src.agent --train

🎮 Run Trained Agent

python -m src.agent

🧪 Hyperparameters
learning_rate: 0.00015
gamma: 0.99
replay_memory_size: 100000
batch_size: 64
epsilon_decay: 0.999995
double_dqn: true
dueling_dqn: true

🛠 Engineering Decisions

Forced CPU execution (stable for small networks)

Reduced network size to prevent overfitting

Increased target sync interval

Lowered learning rate for smoother convergence

Switched from MSE to Huber Loss

📚 Key Learnings

Stabilization techniques matter more than network depth

Double DQN reduces oscillatory Q-values

Dueling improves sparse reward learning

Hyperparameter tuning dominates performance

🔮 Future Improvements

Prioritized Experience Replay

Noisy Networks

Multi-step returns

PPO baseline comparison

Model checkpoint evaluation pipeline
