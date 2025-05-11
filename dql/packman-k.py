import random
import gymnasium as gym
import numpy as np
from collections import deque
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, Conv2D, Flatten
from tensorflow.keras.optimizers import Adam
import time as tt
import cv2
import pickle
import os
import cProfile, pstats, io
import ale_py
gym.register_envs(ale_py)

EPISODES = 100000
PACMAN_ENV = 'MsPacmanDeterministic-v4'


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
        self.model = self._build_model()
        self.state_buffer = deque(maxlen=self.state_size[2])
        for i in range(self.state_size[2]):
            self.state_buffer.append(np.zeros((self.state_size[0], self.state_size[1])))

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        reg = None
        #reg = l2(0.00002)

        model = Sequential()
        conv1 = 96
        convn = 96

        model.add(Conv2D(conv1, input_shape=self.state_size, kernel_size=8, padding="same", activation="relu", strides=(2, 2), kernel_regularizer=reg))
        model.add(Conv2D(convn, kernel_size=5, padding="same", strides=(2, 2), use_bias=False, kernel_regularizer=reg))
        model.add(BatchNormalization())
        model.add(Activation(activation="relu"))
        model.add(Conv2D(convn, kernel_size=3, padding="same", strides=(1, 1), use_bias=False, kernel_regularizer=reg))
        model.add(BatchNormalization())
        model.add(Activation(activation="relu"))
        model.add(Conv2D(convn, kernel_size=3, padding="same", strides=(1, 1), use_bias=False, kernel_regularizer=reg))
        model.add(BatchNormalization())
        model.add(Activation(activation="relu"))
        model.add(Conv2D(convn, kernel_size=3, padding="same", strides=(1, 1), use_bias=False, kernel_regularizer=reg))
        model.add(BatchNormalization())
        model.add(Activation(activation="relu"))
        model.add(Conv2D(convn, kernel_size=3, padding="same", strides=(1, 1), use_bias=False, kernel_regularizer=reg))
        model.add(BatchNormalization())
        model.add(Activation(activation="relu"))
        model.add(Conv2D(convn, kernel_size=3, padding="same", strides=(1, 1), use_bias=False, kernel_regularizer=reg))
        model.add(BatchNormalization())
        model.add(Activation(activation="relu"))
        model.add(Conv2D(convn, kernel_size=3, padding="same", strides=(1, 1), use_bias=False, kernel_regularizer=reg))
        model.add(BatchNormalization())
        model.add(Activation(activation="relu"))
        model.add(Conv2D(convn, kernel_size=3, padding="same", strides=(1, 1), use_bias=False, kernel_regularizer=reg))
        model.add(BatchNormalization())
        model.add(Activation(activation="relu"))
        model.add(Flatten())
        model.add(Dense(512, activation='relu', kernel_regularizer=reg))
        model.add(Dense(self.action_size, activation='linear', kernel_regularizer=reg))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model


    def _build_model2(self):
        # Neural Net for Deep-Q learning Model
        reg = None
        #reg = l2(0.00002)

        model = Sequential()
        conv1 = 96
        convn = 128

        model.add(Conv2D(conv1, input_shape=self.state_size, kernel_size=8, padding="same", strides=(2, 2), activation="relu", kernel_regularizer=reg))
        model.add(Conv2D(convn, kernel_size=5, padding="same", strides=(2, 2), activation='relu', kernel_regularizer=reg))
        model.add(Conv2D(convn, kernel_size=3, padding="same", strides=(1, 1), activation='relu', kernel_regularizer=reg))
        model.add(Conv2D(convn, kernel_size=3, padding="same", strides=(1, 1), activation='relu', kernel_regularizer=reg))
        model.add(Conv2D(convn, kernel_size=3, padding="same", strides=(1, 1), activation='relu', kernel_regularizer=reg))
        model.add(Conv2D(convn, kernel_size=3, padding="same", strides=(1, 1), activation='relu', kernel_regularizer=reg))
        model.add(Conv2D(convn, kernel_size=3, padding="same", strides=(1, 1), activation='relu', kernel_regularizer=reg))
        model.add(Conv2D(convn, kernel_size=3, padding="same", strides=(1, 1), activation='relu', kernel_regularizer=reg))
        model.add(Conv2D(convn, kernel_size=3, padding="same", strides=(1, 1), activation='relu', kernel_regularizer=reg))
        model.add(Flatten())
        model.add(Dense(512, activation='relu', kernel_regularizer=reg))
        model.add(Dense(self.action_size, activation='linear', kernel_regularizer=reg))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done, time):
        self.memory.append((state, action, reward, next_state, done, time))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        states = np.expand_dims(state, axis=0)
        act_values = self.model.predict(states, verbose=0)
        return np.argmax(act_values[0])  # returns action

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
        next_states = []
        states = []
        for state, action, reward, next_state, done, time in batch:
            next_states.append(next_state)
            states.append(state)

        next_states = np.stack(next_states, axis=0)
        states = np.stack(states, axis=0)

        Qn = self.model.predict(next_states, verbose=0)
        Q = self.model.predict(states, verbose=0)

        targets_f = []
        for i, (state, action, reward, next_state, done, time) in enumerate(batch):
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(Qn[i]))
            target_f = Q[i]
            target_f[action] = target
            targets_f.append(target_f)
            # if time == 400:
            #     print("Q: ", Q[i])

        targets_f = np.stack(targets_f, axis=0)
        self.model.fit(states, targets_f, batch_size=mb_size, epochs=1, verbose=0)
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

            next_states = np.stack(next_states, axis=0)
            states = np.stack(states, axis=0)

            Qn = self.model.predict(next_states, verbose=0)
            Q = self.model.predict(states, verbose=0)

            train_set = TrainSet.load_from_file(filename)
            for i, (state, action, reward, next_state, done, time) in enumerate(minibatch):
                target = reward
                if not done:
                    target = (reward + self.gamma * np.amax(Qn[i]))
                target_f = Q[i]
                target_f[action] = target
                train_set.append(state, target_f)

            train_set.save_to_file(filename)

    def fit_train_set(self, train_file, prev_weights_file, weights_file, epochs):
        ts = TrainSet.load_from_file(train_file)
        states, targets = ts.as_arrays()

        if prev_weights_file is not None:
            self.load(prev_weights_file)

        self.model.fit(states, targets,
                       batch_size=256,
                       epochs=epochs,
                       shuffle=True,
                       validation_split=0.02,
                       verbose=2,
                       callbacks=None)

        self.save(weights_file)

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


def create_train_set():
    env = gym.make(PACMAN_ENV, render_mode=None)
    state_size = env.observation_space.shape
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)

    agent.load("./est-pacman_tset.h5")
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
        frame, info = env.reset()
        agent.add_frame_to_buffer(frame)
        state = agent.get_state_buffer()
        score = 0
        for time in range(100000):
            action = agent.act(state)
            next_frame, reward, terminated, truncated, info = env.step(action)
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
    # tr = TrainSet.load_from_file("./trainset.dat")
    # tr.shuffle()
    # tr.save_to_file("./trainset2.dat")
    # print("fin shuffle")
    # tt.sleep(5)
    env = gym.make(PACMAN_ENV)
    state_size = env.observation_space.shape
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    #agent.learning_rate = 0.0001

    agent.fit_train_set("./trainset_pacman.dat", None, "./est-pacman_tset.h5", 15)


def test():
    env = gym.make(PACMAN_ENV, render_mode="human")
    state_size = env.observation_space.shape
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    agent.load("./est-pacman_bn2_new.h5")
    agent.epsilon = 0.0

    for i in range(5):
        frame, info = env.reset()
        agent.add_frame_to_buffer(frame)
        state = agent.get_state_buffer()
        score = 0
        for time in range(10000):
            action = agent.act(state)
            next_frame, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            score += reward
            agent.add_frame_to_buffer(next_frame)
            next_state = agent.get_state_buffer()
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

    env = gym.make(PACMAN_ENV, render_mode=None)
    state_size = env.observation_space.shape
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)

    # agent.load("./est-pacman_98.h5")
    # agent.epsilon_start_decay = 0
    # agent.gamma = 0.98
    # agent.learning_rate = 0.00001
    # agent.epsilon = 0.099  # avg 2000

    # agent.load("./est-pacman_bn2_new.h5")
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
        frame, info = env.reset()
        agent.add_frame_to_buffer(frame)
        state = agent.get_state_buffer()
        score = 0
        for time in range(100000):
            action = agent.act(state)
            next_frame, reward, terminated, truncated, info = env.step(action)
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
            agent.save("./est-pacman_bn2_new.h5")
        if profile:
            pr.disable()
            s = io.StringIO()
            sortby = 'cumulative'
            ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            ps.print_stats()
            print(s.getvalue())
            break

def check_tf_gpu():
    # Set environment variables to help TensorFlow find CUDA
    import os
    # Tell TensorFlow to use the GPU
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    # Set CUDA environment variables
    os.environ['TF_CUDA_PATHS'] = 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3'
    # Print CUDA environment for debugging
    print("CUDA_VISIBLE_DEVICES:", os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set'))
    print("CUDA_PATH:", os.environ.get('CUDA_PATH', 'Not set'))
    
    import tensorflow as tf
    print("TensorFlow version:", tf.__version__)
    
    # Configure GPU memory growth
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            # Set TensorFlow to use the first GPU
            tf.config.set_visible_devices(gpus[0], 'GPU')
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(f"Physical GPUs: {len(gpus)}, Logical GPUs: {len(logical_gpus)}")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    
    print("GPU Available:", tf.config.list_physical_devices('GPU'))
    if tf.config.list_physical_devices('GPU'):
        print("GPU is being used")
        # Print GPU details
        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            print("  Name:", gpu.name)
            print("  Device type:", gpu.device_type)
        # Test GPU computation
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            c = tf.matmul(a, b)
            print("Matrix multiplication result shape:", c.shape)
            print("Computation performed on device:", c.device)
    else:
        print("No GPU available. Using CPU instead.")

if __name__ == "__main__":
    check_tf_gpu()
    #create_train_set()
    #fit_train_set()
    #train()
    #test()

