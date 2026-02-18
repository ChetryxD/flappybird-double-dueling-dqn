🐦 Flappy Bird AI — Double Dueling DQN (PyTorch)

A from-scratch implementation of a Double Dueling Deep Q-Network (DQN) agent trained to master Flappy Bird using value-based reinforcement learning.
The agent learns stable control policies through:
Experience Replay

Target Network Synchronization

Double DQN (overestimation reduction)

Dueling Architecture (separate value & advantage streams)

Huber Loss stabilization


After training, the agent consistently achieves:

✅ 50+ pipes passed

📈 Stable learning curve

🎯 Strong generalization beyond early episodes

🚀 Project Highlights

Built modular RL training pipeline in PyTorch

Implemented Double DQN to reduce Q-value overestimation

Implemented Dueling Network Architecture for better state-value learning

Tuned hyperparameters for stable long-duration training

Achieved strong performance in sparse reward environment


🧠 Architecture:
| Technique            | Purpose                                    |
| -------------------- | ------------------------------------------ |
| Experience Replay    | Breaks correlation between samples         |
| Target Network       | Stabilizes Q-learning updates              |
| Double DQN           | Reduces Q-value overestimation             |
| Dueling DQN          | Separates state value and action advantage |
| SmoothL1Loss (Huber) | Stabilizes training                        |


📊 Results
Training duration: ~3–4 hours
Convergence after ~300k episodes
Peak score: 120+ pipes
Consistent average: 40–70 pipes

Training curve example:
runs/flappybird1.png

📂 Project Structure:

dqn_pytorch/

src/
    agent.py
   
    dqn.py
    
    experience_replay.py

configs/
    
    hyperparameters.yml

assets/
   
    learning_curve.png
    
    flappybird_demo.gif

runs/               (training outputs - ignored in git)

requirements.txt

README.md

⚙️ Installation:

Clone the repository:

git clone https://github.com/ChetryxD/flappybird-double-dueling-dqn.git

cd flappybird-double-dueling-dqn


Create virtual environment (recommended):

python -m venv venv

venv\Scripts\activate   # Windows


Install dependencies:

pip install -r requirements.txt


🏋️ Training

Run training:

python -m src.agent --train

Training logs and model will be saved in:

runs/


Saved artifacts:

flappybird1.pt → trained model

flappybird1.log → training log

flappybird1.png → learning curve


🎮 Run Trained Agent

After training:

python -m src.agent


🧪 Hyperparameters Used

Located in:

configs/hyperparameters.yml



Key configuration:

Learning rate: 0.00015

Discount factor: 0.99

Replay memory: 100,000

Batch size: 64

Epsilon decay: 0.999995

Double DQN: Enabled

Dueling DQN: Enabled



🛠 Key Engineering Decisions

Forced CPU execution (more stable for small network workloads)

Reduced network size to prevent overfitting

Increased target sync interval for smoother updates

Lowered learning rate for long-term stability

Switched from MSE to Huber loss



📈 What I Learned

Stabilization techniques matter more than raw architecture size

Hyperparameter tuning significantly impacts convergence

Dueling architecture improves value estimation in sparse reward settings

Double DQN reduces unstable oscillations



📌 Future Improvements

Prioritized Experience Replay

Noisy Networks for exploration

Multi-step returns

Policy gradient comparison (PPO baseline)
