# -*- coding: utf-8 -*-
import random
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPISODES = 10000

class DQNNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)  # Increased layer size
        self.fc2 = nn.Linear(64, 64)  # Increased layer size
        self.fc3 = nn.Linear(64, action_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)  # Increased memory size
        self.gamma = 0.99    # Increased discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999  # Slower decay for better exploration
        self.learning_rate = 0.0005  # Adjusted learning rate
        self.model = self._build_model()
        self.target_model = self._build_model()  # Target network
        self.update_target_model()  # Copy weights to target model
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)  # Added weight decay
        self.criterion = nn.MSELoss()
        self.update_target_counter = 0
        self.update_target_frequency = 10  # Update target model every N episodes

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = DQNNetwork(self.state_size, self.action_size).to(device)
        return model
        
    def update_target_model(self):
        # Copy weights from model to target_model
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done, time):
        self.memory.append((state, action, reward, next_state, done, time))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.FloatTensor(state).to(device)
        with torch.no_grad():
            act_values = self.model(state_tensor)
        return torch.argmax(act_values[0]).item()  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        next_states = []
        states = []
        for state, action, reward, next_state, done, time in minibatch:
            next_states.append(next_state[0])
            states.append(state[0])

        next_states = np.stack(next_states, axis=0)
        states = np.stack(states, axis=0)
        
        next_states_tensor = torch.FloatTensor(next_states).to(device)
        states_tensor = torch.FloatTensor(states).to(device)
        
        with torch.no_grad():
            # Use target model for more stable Q values
            Qn = self.target_model(next_states_tensor).cpu().numpy()
            Q = self.model(states_tensor).cpu().numpy()

        targets_f = []
        for i, (state, action, reward, next_state, done, time) in enumerate(minibatch):
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(Qn[i]))
            target_f = Q[i]
            target_f[action] = target
            targets_f.append(target_f)

        targets_f = np.stack(targets_f, axis=0)
        targets_tensor = torch.FloatTensor(targets_f).to(device)
        
        self.optimizer.zero_grad()
        outputs = self.model(states_tensor)
        loss = self.criterion(outputs, targets_tensor)
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_state_dict(torch.load(name))
        self.model.eval()

    def save(self, name):
        torch.save(self.model.state_dict(), name)


def test():
    env = gym.make('CartPole-v1', render_mode="human")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    agent.load("./est-dqn_best.pt")
    agent.epsilon = 0.01

    state, _ = env.reset()
    state = np.reshape(state, [1, state_size])
    for time in range(1000):
        env.render()
        action = agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = np.reshape(next_state, [1, state_size])
        state = next_state
        if done:
            print("score: {}".format(time))
            break


def train():
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    # agent.load("./est-dqn.pt")
    # agent.epsilon = 0.01
    batch_size = 128  # Increased batch size
    score_window = deque(maxlen=100)
    best_score = 0

    for e in range(EPISODES):
        state, _ = env.reset()
        state = np.reshape(state, [1, state_size])
        total_reward = 0
        
        for time in range(1000):
            # env.render()
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Better reward shaping
            reward = 1.0  # Reward for surviving each step
            
            # Add position penalty - penalize being away from center
            # Since we need to access the raw state before reshaping
            cart_position = next_state[0]  # Get the cart position (first element)
            position_penalty = 0.1 * abs(cart_position)  # Penalty proportional to distance from center
            reward -= position_penalty  # Subtract penalty from reward
            
            if done and time < 499:
                reward = -10 * (1 - time/499)  # Proportional penalty for early termination
            if time >= 499:
                reward = 20  # Bonus for reaching the target
                
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done, time)
            state = next_state
            total_reward += reward
            
            if done:
                # Update target network periodically
                agent.update_target_counter += 1
                if agent.update_target_counter >= agent.update_target_frequency:
                    agent.update_target_model()
                    agent.update_target_counter = 0
                
                score_window.append(time)
                score_avg = sum(score_window) / len(score_window)
                
                # Save best model
                if score_avg > best_score and len(score_window) == 100:
                    best_score = score_avg
                    agent.save("./est-dqn_best.pt")
                    print("New best model saved with avg score: {:.1f}".format(best_score))
                
                print("episode: {}/{}, score: {}, avg: {:.1f}, e: {:.2}"
                      .format(e, EPISODES, time, score_avg, agent.epsilon))
                break
                
        # Training with more frequent updates
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
                
        if e % 100 == 0:
            agent.save("./est-dqn.pt")


if __name__ == "__main__":
   #train()
   test()

