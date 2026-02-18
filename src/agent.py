import os
import argparse
import itertools
from datetime import datetime, timedelta

import gymnasium as gym
import numpy as np
import torch
from torch import nn
import yaml
import matplotlib
import matplotlib.pyplot as plt

from src.experience_replay import ReplayMemory
from src.dqn import DQN

import flappy_bird_gymnasium

matplotlib.use("Agg")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===============================
# Robust Project Paths
# ===============================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "configs", "hyperparameters.yml")
RUNS_DIR = os.path.join(BASE_DIR, "runs")
os.makedirs(RUNS_DIR, exist_ok=True)


# ===============================
# Agent
# ===============================

class Agent:

    def __init__(self, hyperparameter_set="flappybird1"):

        with open(CONFIG_PATH, "r") as file:
            all_sets = yaml.safe_load(file)
            hp = all_sets[hyperparameter_set]

        self.hp_name = hyperparameter_set

        # Hyperparameters
        self.env_id = hp["env_id"]
        self.learning_rate = hp["learning_rate_a"]
        self.gamma = hp["discount_factor_g"]
        self.memory_size = hp["replay_memory_size"]
        self.batch_size = hp["mini_batch_size"]
        self.epsilon = hp["epsilon_init"]
        self.epsilon_decay = hp["epsilon_decay"]
        self.epsilon_min = hp["epsilon_min"]
        self.fc1_nodes = hp["fc1_nodes"]
        self.enable_double = hp["enable_double_dqn"]
        self.enable_dueling = hp["enable_dueling_dqn"]
        self.env_make_params = hp.get("env_make_params", {})

        # Files
        self.model_file = os.path.join(RUNS_DIR, f"{self.hp_name}.pt")
        self.log_file = os.path.join(RUNS_DIR, f"{self.hp_name}.log")
        self.graph_file = os.path.join(RUNS_DIR, f"{self.hp_name}.png")

        # Loss
        self.loss_fn = nn.SmoothL1Loss()

    # ===================================
    # RUN
    # ===================================

    def run(self, is_training=True, render=False):

        env = gym.make(self.env_id,
                       render_mode="human" if render else None,
                       **self.env_make_params)

        n_actions = env.action_space.n
        n_states = env.observation_space.shape[0]

        policy_net = DQN(n_states, n_actions,
                         self.fc1_nodes,
                         self.enable_dueling).to(device)

        if is_training:
            target_net = DQN(n_states, n_actions,
                             self.fc1_nodes,
                             self.enable_dueling).to(device)
            target_net.load_state_dict(policy_net.state_dict())

            optimizer = torch.optim.Adam(policy_net.parameters(),
                                         lr=self.learning_rate)

            memory = ReplayMemory(self.memory_size)

            rewards_history = []
            epsilon_history = []
            best_reward = -float("inf")

            print("Training starting...")

        else:
            if not os.path.exists(self.model_file):
                print("Model not found.")
                return

            policy_net.load_state_dict(torch.load(self.model_file, map_location=device))
            policy_net.eval()

        # ===========================
        # EPISODES
        # ===========================

        for episode in itertools.count():

            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device)

            terminated = False
            truncated = False
            total_reward = 0

            while not (terminated or truncated):

                # Epsilon-greedy
                if is_training and np.random.rand() < self.epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        action = policy_net(state.unsqueeze(0)).argmax().item()

                new_state, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward

                new_state = torch.tensor(new_state,
                                         dtype=torch.float32,
                                         device=device)

                if is_training:
                    memory.append((state,
                                   action,
                                   new_state,
                                   reward,
                                   terminated))

                    if len(memory) > self.batch_size:
                        self.optimize(memory, policy_net, target_net, optimizer)

                    # Epsilon decay
                    self.epsilon = max(self.epsilon * self.epsilon_decay,
                                       self.epsilon_min)

                state = new_state

            if is_training:

                rewards_history.append(total_reward)
                epsilon_history.append(self.epsilon)

                if total_reward > best_reward:
                    best_reward = total_reward
                    torch.save(policy_net.state_dict(), self.model_file)
                    print(f"New best reward: {best_reward:.2f}")

                if episode % 1000 == 0:
                    self.save_graph(rewards_history, epsilon_history)

            else:
                print(f"Reward: {total_reward}")

    # ===================================
    # OPTIMIZATION
    # ===================================

    def optimize(self, memory, policy_net, target_net, optimizer):

        batch = memory.sample(self.batch_size)

        states, actions, next_states, rewards, dones = zip(*batch)

        states = torch.stack(states)
        actions = torch.tensor(actions, device=device)
        next_states = torch.stack(next_states)
        rewards = torch.tensor(rewards, device=device)
        dones = torch.tensor(dones, dtype=torch.float32, device=device)

        # Current Q
        current_q = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()

        # Target Q
        with torch.no_grad():
            if self.enable_double:
                best_actions = policy_net(next_states).argmax(1)
                target_q = target_net(next_states).gather(
                    1, best_actions.unsqueeze(1)).squeeze()
            else:
                target_q = target_net(next_states).max(1)[0]

            target = rewards + (1 - dones) * self.gamma * target_q

        loss = self.loss_fn(current_q, target)

        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)

        optimizer.step()

        # Soft update
        tau = 0.005
        for target_param, policy_param in zip(target_net.parameters(),
                                              policy_net.parameters()):
            target_param.data.copy_(
                tau * policy_param.data +
                (1.0 - tau) * target_param.data
            )

    # ===================================
    # SAVE GRAPH
    # ===================================

    def save_graph(self, rewards, epsilons):

        fig = plt.figure()

        mean_rewards = np.convolve(rewards,
                                   np.ones(100)/100,
                                   mode="valid")

        plt.subplot(121)
        plt.title("Mean Rewards")
        plt.plot(mean_rewards)

        plt.subplot(122)
        plt.title("Epsilon")
        plt.plot(epsilons)

        fig.savefig(self.graph_file)
        plt.close(fig)


# ===============================
# ENTRY POINT
# ===============================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()

    agent = Agent()

    if args.train:
        agent.run(is_training=True)
    else:
        agent.run(is_training=False, render=True)