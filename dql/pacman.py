# -*- coding: utf-8 -*-
import random
import gymnasium as gym
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time as tt
import matplotlib.pyplot as plt
import cv2
import pickle
import os
import cProfile, pstats, io
import ale_py
gym.register_envs(ale_py)

# Set matmul precision for better performance on compatible GPUs
if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7: # Ampere and newer
    torch.set_float32_matmul_precision('high')

EPISODES = 100000


class TrainSet:
    def __init__(self):
        self.records = []

    def append(self, state, target):
        self.records.append((state, target))

    def save_to_file(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    def shuffle(self):
        random.shuffle(self.records)

    def as_arrays(self):
        states = list(map(lambda r: r[0], self.records))
        targets = list(map(lambda r: r[1], self.records))
        return np.stack(states), np.stack(targets)

    @classmethod
    def load_from_file(cls, filename) -> 'TrainSet':
        if not os.path.isfile(filename):
            return TrainSet()
        with open(filename, "rb") as f:
            return pickle.load(f)


class DQNModel(nn.Module):
    def __init__(self, state_size, action_size, reg=None):
        super(DQNModel, self).__init__()
        self.conv1 = nn.Conv2d(state_size[2], 96, kernel_size=8, stride=2, padding=3)
        self.conv2 = nn.Conv2d(96, 96, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(96)
        self.conv3 = nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(96)
        self.conv4 = nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(96)
        self.conv5 = nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(96)
        self.conv6 = nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(96)
        self.conv7 = nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(96)
        self.conv8 = nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn8 = nn.BatchNorm2d(96)
        
        # Calculate the flattened size based on input dimensions
        self._to_linear = None
        self._get_conv_output(state_size)
        
        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, action_size)
        
    def _get_conv_output(self, shape):
        bs = 1
        input = torch.rand(bs, shape[2], shape[0], shape[1])
        output = self._forward_conv(input)
        self._to_linear = int(np.prod(output.size()))
        
    def _forward_conv(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn2(self.conv2(x))
        x = F.relu(x)
        x = self.bn3(self.conv3(x))
        x = F.relu(x)
        x = self.bn4(self.conv4(x))
        x = F.relu(x)
        x = self.bn5(self.conv5(x))
        x = F.relu(x)
        x = self.bn6(self.conv6(x))
        x = F.relu(x)
        x = self.bn7(self.conv7(x))
        x = F.relu(x)
        x = self.bn8(self.conv8(x))
        x = F.relu(x)
        return x
        
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # Convert from NHWC to NCHW format
        x = self._forward_conv(x)
        # Replace view with reshape or make the tensor contiguous first
        x = x.contiguous().view(x.size(0), -1)  # Make tensor contiguous before using view
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DQNAgent:
    def __init__(self, state_size, action_size):
        # self.state_size = (210, 160, 6)
        self.state_size = (105, 80, 6)
        self.action_size = action_size
        self.memory = deque(maxlen=50000)
        self.gamma = 0.99    # discount rate
        self.replay_count = 0
        self.epsilon_start_decay = 5000
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9999
        self.learning_rate = 0.00001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model().to(self.device)
        # Compile the model for potential performance improvement
        # Use "aot_eager" backend to avoid Triton dependency if it's causing issues
        try:
            self.model = torch.compile(self.model, backend="aot_eager")
        except Exception as e:
            print(f"Failed to compile model with aot_eager: {e}")
            print("Proceeding without model compilation.")
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.state_buffer = deque(maxlen=self.state_size[2])
        for i in range(self.state_size[2]):
            self.state_buffer.append(np.zeros((self.state_size[0], self.state_size[1])))

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        return DQNModel(self.state_size, self.action_size)

    def remember(self, state, action, reward, next_state, done, time):
        self.memory.append((state, action, reward, next_state, done, time))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.FloatTensor(np.expand_dims(state, axis=0)).to(self.device)
        with torch.no_grad():
            act_values = self.model(state_tensor)
        return torch.argmax(act_values[0]).item()  # returns action

    def reset_state_buffer(self):
        for i in range(self.state_size[2]):
            self.state_buffer.append(np.zeros((self.state_size[0], self.state_size[1])))

    def add_frame_to_buffer(self, frame):
        frame = cv2.resize(frame, (self.state_size[1], self.state_size[0]), interpolation=cv2.INTER_AREA)
        frame = (frame / 255).astype('float32')
        frame = [frame[:, :, i] for i in range(frame.shape[2])]
        self.state_buffer.extend(frame)

    def get_state_buffer(self):
        state_buffer = np.stack(self.state_buffer)
        state_buffer = np.transpose(state_buffer, (1, 2, 0))
        return state_buffer

    def replay(self, batch_size):
        self.replay_count += 1
        mb_size = 256
        batch_size = batch_size - (batch_size % mb_size)
        batch = random.sample(self.memory, batch_size)
        
        for i in range(0, batch_size, mb_size):
            mini_batch = batch[i:i+mb_size]
            states = np.array([b[0] for b in mini_batch])
            actions = np.array([b[1] for b in mini_batch])
            rewards = np.array([b[2] for b in mini_batch])
            next_states = np.array([b[3] for b in mini_batch])
            dones = np.array([b[4] for b in mini_batch])
            
            states_tensor = torch.FloatTensor(states).to(self.device)
            next_states_tensor = torch.FloatTensor(next_states).to(self.device)
            actions_tensor = torch.LongTensor(actions).to(self.device)
            rewards_tensor = torch.FloatTensor(rewards).to(self.device)
            dones_tensor = torch.FloatTensor(dones).to(self.device)
            
            # Get current Q values
            current_q_values = self.model(states_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
            
            # Get next Q values
            with torch.no_grad():
                next_q_values = self.model(next_states_tensor).max(1)[0]
                target_q_values = rewards_tensor + (1 - dones_tensor) * self.gamma * next_q_values
            
            # Compute loss
            loss = F.mse_loss(current_q_values, target_q_values)
            
            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        if self.replay_count > self.epsilon_start_decay and self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_train_set(self, batch_size, filename):
        while len(self.memory) > 0:
            minibatch = [self.memory.popleft() for _i in range(min(batch_size, len(self.memory)))]
            next_states = []
            states = []
            for state, action, reward, next_state, done, time in minibatch:
                next_states.append(next_state)
                states.append(state)

            next_states_tensor = torch.FloatTensor(np.stack(next_states, axis=0)).to(self.device)
            states_tensor = torch.FloatTensor(np.stack(states, axis=0)).to(self.device)

            with torch.no_grad():
                Qn = self.model(next_states_tensor).cpu().numpy()
                Q = self.model(states_tensor).cpu().numpy()

            train_set = TrainSet.load_from_file(filename)
            for i, (state, action, reward, next_state, done, time) in enumerate(minibatch):
                target = reward
                if not done:
                    target = (reward + self.gamma * np.amax(Qn[i]))
                target_f = Q[i].copy()
                target_f[action] = target
                train_set.append(state, target_f)

            train_set.save_to_file(filename)

    def fit_train_set(self, train_file, prev_weights_file, weights_file, epochs):
        ts = TrainSet.load_from_file(train_file)
        states, targets = ts.as_arrays()

        if prev_weights_file is not None:
            self.load(prev_weights_file)

        states_tensor = torch.FloatTensor(states).to(self.device)
        targets_tensor = torch.FloatTensor(targets).to(self.device)
        
        dataset = torch.utils.data.TensorDataset(states_tensor, targets_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)
        
        validation_split = 0.02
        dataset_size = len(dataset)
        val_size = int(validation_split * dataset_size)
        train_size = dataset_size - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=256, shuffle=False)
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            for batch_states, batch_targets in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_states)
                loss = F.mse_loss(outputs, batch_targets)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_states, batch_targets in val_loader:
                    outputs = self.model(batch_states)
                    val_loss += F.mse_loss(outputs, batch_targets).item()
            
            print(f"Epoch {epoch+1}/{epochs}, Training loss: {train_loss/len(train_loader):.6f}, Validation loss: {val_loss/len(val_loader):.6f}")

        self.save(weights_file)

    def load(self, name):
        self.model.load_state_dict(torch.load(name))
        self.model.eval()

    def save(self, name):
        torch.save(self.model.state_dict(), name)


def create_train_set():
    env = gym.make('ALE/MsPacman-v5', render_mode=None)
    state_size = env.observation_space.shape
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)

    agent.load("./est-pacman_h.pt")
    agent.epsilon_start_decay = 0
    agent.gamma = 0.99
    agent.learning_rate = 0.00005
    agent.epsilon = 0.01
    agent.epsilon_min = 0.01

    batch_size = 512
    score_window = deque(maxlen=100)

    EPISODES = 50

    for e in range(EPISODES):
        agent.reset_state_buffer()
        frame, _ = env.reset()
        agent.add_frame_to_buffer(frame)
        state = agent.get_state_buffer()
        score = 0
        for time in range(100000):
            action = agent.act(state)
            next_frame, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.add_frame_to_buffer(next_frame)
            next_state = agent.get_state_buffer()

            score += reward

            # if done:
            #     reward = -250

            agent.remember(state, action, reward, next_state, done, time)
            state = next_state
            if done:
                # if e == 0:
                #     x = state[:,:,0]
                #     plt.imshow(x)
                #     plt.show()
                score_window.append(score)
                score_avg = sum(score_window) / len(score_window)
                print("episode: {}/{}, score: {}, avg: {:.1f}, frames: {}, e: {:.2}"
                      .format(e, EPISODES, score, score_avg, time, agent.epsilon))
                break
        if len(agent.memory) > batch_size:
            agent.save_train_set(batch_size, "trainset_pacman.dat")


def fit_train_set():
    env = gym.make('ALE/MsPacman-v5', render_mode=None)
    state_size = env.observation_space.shape
    action_size = env.observation_space.n
    agent = DQNAgent(state_size, action_size)

    agent.fit_train_set("./trainset_pacman.dat", None, "./est-pacman_default.pt", 15)


def test():
    env = gym.make("ALE/MsPacman-v5", render_mode="human")
    state_size = env.observation_space.shape
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    agent.load("./est-pacman_bn2.pt")
    agent.epsilon = 0.0

    for i in range(5):
        frame, _ = env.reset()
        agent.add_frame_to_buffer(frame)
        state = agent.get_state_buffer()
        score = 0
        for time in range(10000):
            action = agent.act(state)
            next_frame, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            score += reward
            agent.add_frame_to_buffer(next_frame)
            next_state = agent.get_state_buffer()
            tt.sleep(0.02)
            # scipy.misc.imsave(f'frame{time:0=3d}.jpg', state[:, :, 0:3])
            state = next_state
            if time % 100 == 0:
                print("frame: ", time)
            if done:
                # x = state[:,:,0]
                # plt.imshow(x)
                # plt.show()
                print("score: {}, frames:{}".format(score, time))
                break


def train():
    env = gym.make('ALE/MsPacman-v5', render_mode=None)
    state_size = env.observation_space.shape
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    
    #agent.load("./est-pacman_bn2.pt")

    # agent.load("./est-pacman_98.pt")
    # agent.epsilon_start_decay = 0
    # agent.gamma = 0.98
    # agent.learning_rate = 0.00001
    # agent.epsilon = 0.099  # avg 2000

    # agent.load("./est-pacman_bn2.pt")
    # agent.epsilon_start_decay = 0
    # agent.gamma = 0.98
    # agent.learning_rate = 0.00002
    # agent.epsilon = 0.1
    # agent.epsilon_min = 0.0

    min_batch_size = 512
    score_window = deque(maxlen=100)
    frames_window = deque(maxlen=100)

    for e in range(EPISODES):
        profile = False and e == 50
        if profile:
            pr = cProfile.Profile()
            pr.enable()

        agent.reset_state_buffer()
        frame, _ = env.reset()
        agent.add_frame_to_buffer(frame)
        state = agent.get_state_buffer()
        score = 0
        for time in range(100000):
            action = agent.act(state)
            next_frame, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.add_frame_to_buffer(next_frame)
            next_state = agent.get_state_buffer()

            score += reward

            if done:
                reward = -50

            agent.remember(state, action, reward, next_state, done, time)
            state = next_state
            if done:
                score_window.append(score)
                frames_window.append(time)
                score_avg = sum(score_window) / len(score_window)
                frames_avg = int(sum(frames_window) / len(frames_window))
                print("episode: {}/{}, score: {}, avg: {:.1f}, frames: {}, e: {:.2}"
                      .format(e, EPISODES, score, score_avg, time, agent.epsilon))
                break
        if len(agent.memory) > 20000:
            agent.replay(max(min_batch_size, frames_avg))
        if e != 0 and e % 100 == 0:
            agent.save("./est-pacman_bn2.pt")
        if profile:
            pr.disable()
            s = io.StringIO()
            sortby = 'cumulative'
            ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            ps.print_stats()
            print(s.getvalue())
            break


def test_gpu_performance():
    import torch
    import time
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. Running on CPU.")
        return False
    
    # Get device information
    device = torch.device("cuda")
    print(f"Using device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    
    # Create large tensors to test computation speed
    size = 5000
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    
    # Warm-up run
    torch.matmul(a, b)
    torch.cuda.synchronize()
    
    # Benchmark matrix multiplication
    iterations = 10
    start_time = time.time()
    for _ in range(iterations):
        c = torch.matmul(a, b)
        torch.cuda.synchronize()  # Wait for GPU to finish
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    ops_per_second = iterations / elapsed_time
    print(f"Performed {iterations} matrix multiplications in {elapsed_time:.4f} seconds")
    print(f"Performance: {ops_per_second:.2f} operations per second")
    
    # Check if performance seems reasonable for a GPU
    # An RTX 5070 should easily do multiple ops per second for this size
    is_gpu_speed = ops_per_second > 1.0
    
    # Check memory usage
    print(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    print(f"GPU memory reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    
    if is_gpu_speed:
        print("Performance indicates GPU is being utilized properly.")
    else:
        print("Performance is lower than expected. GPU might not be utilized properly.")
    
    return is_gpu_speed

if __name__ == "__main__":
    # gpu_working = test_gpu_performance()
    
    # if gpu_working:
    #     print("GPU is working well, proceeding with training")
    # else:
    #     print("WARNING: GPU performance is not optimal")
    
    #create_train_set()
    #fit_train_set()
    train()
    #test()

