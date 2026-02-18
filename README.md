## 🎮 Demo
![flappybirds](https://github.com/user-attachments/assets/66528b50-d6ea-4e2f-a4d0-ac5f4d110708)




# 🐦 Flappy Bird AI — Double Dueling DQN (PyTorch)

A from-scratch implementation of a **Double Dueling Deep Q-Network (DQN)** agent that masters Flappy Bird using value-based reinforcement learning.

---

## 📌 Overview

This project implements a **Double Dueling Deep Q-Network** trained to solve Flappy Bird in a sparse-reward environment.

The agent achieves stable, long-horizon control by combining:

- Experience Replay  
- Target Network Synchronization  
- Double DQN (overestimation mitigation)  
- Dueling Network Architecture  
- Huber Loss stabilization  

The result is consistent high-performance gameplay with strong generalization beyond early training episodes.

---

## 🚀 Performance

| Metric | Value |
|--------|-------|
| Peak Score | 80+ pipes |
| Consistent Average | 20–50 pipes |
| Convergence | ~300k episodes |
| Training Time | ~4–6 hours (CPU) |

✅ Stable learning curve  
✅ Reduced Q-value oscillations  
✅ Generalization beyond memorized sequences  

---

## 🧠 Algorithmic Architecture

| Technique | Why It Matters |
|------------|----------------|
| Experience Replay | Breaks temporal correlation |
| Target Network | Stabilizes bootstrapped updates |
| Double DQN | Reduces Q-value overestimation bias |
| Dueling Architecture | Separates state value from action advantage |
| SmoothL1Loss (Huber) | Prevents gradient explosions |

---

## Dueling Network Structure

Q(s, a) is decomposed as:

```python
Q(s, a) = V(s) + (A(s, a) - mean(A(s, ·)))\

📈 Training Curve
<img width="788" height="587" alt="image" src="https://github.com/user-attachments/assets/4962cf31-4cf1-40cd-97b0-5aca20594e53" />

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
