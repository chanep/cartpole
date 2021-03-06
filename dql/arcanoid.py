# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Activation, Conv2D, Flatten
from keras.optimizers import Adam
import time as tt
from keras.regularizers import l2
import matplotlib.pyplot as plt
import scipy.misc
import cv2
import pickle
import os

from dql.residual import Residual

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


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = (105, 80, 4)
        self.action_size = action_size
        self.memory = deque(maxlen=100000)
        self.gamma = 0.97    # discount rate
        self.replay_count = 0
        self.epsilon_start_decay = 10000
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.003
        self.epsilon_decay = 0.9999
        self.learning_rate = 0.00002
        self.model = self._build_model()
        self.state_buffer = deque(maxlen=self.state_size[2])
        for i in range(self.state_size[2]):
            self.state_buffer.append(np.zeros((self.state_size[0], self.state_size[1])))


    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        reg = None
        # reg = l2(2e-5)

        model = Sequential()

        model.add(Conv2D(96, input_shape=self.state_size, kernel_size=8, padding="same", strides=(2, 2), activation="relu", kernel_regularizer=reg))
        model.add(Conv2D(128, kernel_size=5, padding="same", strides=(2, 2), activation='relu', kernel_regularizer=reg))
        # model.add(Residual(128, (3, 3)))
        model.add(Conv2D(128, kernel_size=3, padding="same", strides=(1, 1), activation='relu', kernel_regularizer=reg))
        model.add(Conv2D(128, kernel_size=3, padding="same", strides=(1, 1), activation='relu', kernel_regularizer=reg))
        model.add(Conv2D(128, kernel_size=3, padding="same", strides=(1, 1), activation='relu', kernel_regularizer=reg))
        model.add(Conv2D(128, kernel_size=3, padding="same", strides=(1, 1), activation='relu', kernel_regularizer=reg))
        model.add(Conv2D(128, kernel_size=3, padding="same", strides=(1, 1), activation='relu', kernel_regularizer=reg))
        model.add(Conv2D(128, kernel_size=3, padding="same", strides=(1, 1), activation='relu', kernel_regularizer=reg))
        model.add(Conv2D(128, kernel_size=3, padding="same", strides=(1, 1), activation='relu', kernel_regularizer=reg))
        model.add(Flatten())
        model.add(Dense(512, activation='relu', kernel_regularizer=reg))
        # model.add(Dense(128, activation='relu', kernel_regularizer=reg))
        model.add(Dense(self.action_size, activation='linear', kernel_regularizer=reg))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done, time):
        self.memory.append((state, action, reward, next_state, done, time))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        states = np.expand_dims(state, axis=0)
        act_values = self.model.predict(states)
        return np.argmax(act_values[0])  # returns action

    def reset_state_buffer(self):
        for i in range(self.state_size[2]):
            self.state_buffer.append(np.zeros((self.state_size[0], self.state_size[1])))

    def add_frame_to_buffer(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.state_size[1], self.state_size[0]), interpolation=cv2.INTER_AREA)
        monocrome_state = frame[:, :]
        #print("monocrome shape: ", monocrome_state.shape)
        self.state_buffer.append(monocrome_state)

    def get_state_buffer(self):
        state_buffer = np.stack(self.state_buffer)
        state_buffer = np.transpose(state_buffer, (1, 2, 0))
        return state_buffer

    def replay(self, batch_size):
        self.replay_count += 1
        minibatch = random.sample(self.memory, batch_size)
        next_states = []
        states = []
        for state, action, reward, next_state, done, time in minibatch:
            next_states.append(next_state)
            states.append(state)

        next_states = np.stack(next_states, axis=0)
        states = np.stack(states, axis=0)

        Qn = self.model.predict(next_states)
        Q = self.model.predict(states)

        targets_f = []
        for i, (state, action, reward, next_state, done, time) in enumerate(minibatch):
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(Qn[i]))
            target_f = Q[i]
            target_f[action] = target
            targets_f.append(target_f)
            # if time == 400:
            #     print("Q: ", Q[i])

        targets_f = np.stack(targets_f, axis=0)
        self.model.fit(states, targets_f, batch_size=256, epochs=1, verbose=0)
        if self.replay_count > self.epsilon_start_decay and self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_train_set(self, batch_size, filename):
        while len(self.memory) > 0:
            minibatch = [self.memory.popleft() for _i in range(max(batch_size, len(self.memory)))]
            next_states = []
            states = []
            for state, action, reward, next_state, done, time in minibatch:
                next_states.append(next_state)
                states.append(state)

            next_states = np.stack(next_states, axis=0)
            states = np.stack(states, axis=0)

            Qn = self.model.predict(next_states)
            Q = self.model.predict(states)

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


def test():
    env = gym.make("Breakout-v0")
    state_size = env.observation_space.shape
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    agent.load("./est-breakout_7conv.h5")
    agent.epsilon = 0.0

    frame = env.reset()
    agent.add_frame_to_buffer(frame)
    state = agent.get_state_buffer()
    score = 0
    for time in range(10000):
        env.render()
        tt.sleep(0.02)
        action = agent.act(state)
        next_frame, reward, done, _ = env.step(action)
        score += reward
        agent.add_frame_to_buffer(next_frame)
        next_state = agent.get_state_buffer()
        # scipy.misc.imsave(f'frame{time:0=3d}.jpg', state[:,:,0])
        state = next_state
        if time % 10 == 0:
            print("frame: ", time)
        if done:
            # x = state[:,:,0]
            # plt.imshow(x)
            # plt.show()
            print("score: {}, frames:{}".format(score, time))
            break


def create_train_set():
    env = gym.make('Breakout-v0')
    state_size = env.observation_space.shape
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)

    agent.load("./est-breakout_22.h5")
    agent.epsilon_start_decay = 0
    agent.gamma = 0.99
    agent.learning_rate = 0.00002
    agent.epsilon = 0.003
    agent.epsilon_min = 0.003

    batch_size = 512
    score_window = deque(maxlen=100)

    EPISODES = 50

    for e in range(EPISODES):
        agent.reset_state_buffer()
        frame = env.reset()
        agent.add_frame_to_buffer(frame)
        state = agent.get_state_buffer()
        score = 0
        for time in range(100000):
            action = agent.act(state)
            next_frame, reward, done, _ = env.step(action)
            agent.add_frame_to_buffer(next_frame)
            next_state = agent.get_state_buffer()

            score += reward

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
            agent.save_train_set(batch_size, "trainset.dat")


def fit_train_set():
    # tr = TrainSet.load_from_file("./trainset.dat")
    # tr.shuffle()
    # tr.save_to_file("./trainset2.dat")
    # print("fin shuffle")
    # tt.sleep(5)
    env = gym.make('Breakout-v0')
    state_size = env.observation_space.shape
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)

    agent.fit_train_set("./trainset.dat", None, "./est-breakout_6res.h5", 15)


def train():
    env = gym.make('Breakout-v0')
    state_size = env.observation_space.shape
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)

    # agent.load("./est-breakout_7conv.h5")
    # agent.epsilon_start_decay = 0
    # agent.epsilon = 0.057
    # agent.learning_rate = 0.00002

    min_batch_size = 512
    score_window = deque(maxlen=100)
    frames_window = deque(maxlen=100)

    for e in range(EPISODES):
        agent.reset_state_buffer()
        frame = env.reset()
        agent.add_frame_to_buffer(frame)
        state = agent.get_state_buffer()
        score = 0
        for time in range(100000):
            action = agent.act(state)
            next_frame, reward, done, _ = env.step(action)
            agent.add_frame_to_buffer(next_frame)
            next_state = agent.get_state_buffer()

            score += reward

            agent.remember(state, action, reward, next_state, done, time)
            state = next_state
            if done:
                # if e == 0:
                #     x = state[:,:,0]
                #     plt.imshow(x)
                #     plt.show()
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
            agent.save("./est-breakout_7conv.h5")


if __name__ == "__main__":
    #create_train_set()
    #fit_train_set()
    #train()
    test()

