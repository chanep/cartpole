# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Activation, Conv2D, Flatten
from keras.optimizers import Adam
from keras.regularizers import l2
import matplotlib.pyplot as plt
import cv2

EPISODES = 10000

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = (84, 84, 4)
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.learning_rate = 0.0002
        self.model = self._build_model()
        self.state_buffer = deque(maxlen=self.state_size[2])
        for i in range(self.state_size[2]):
            self.state_buffer.append(np.zeros((self.state_size[0], self.state_size[1])))


    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        reg = None
        # reg = l2(0.0001)

        model = Sequential()

        model.add(Conv2D(64, input_shape=self.state_size, kernel_size=8, padding="same", strides=(4, 4), activation="relu"))
        model.add(Conv2D(64, kernel_size=5, padding="same", strides=(2, 2), activation='relu'))
        model.add(Conv2D(64, kernel_size=3, padding="same", strides=(1, 1), activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
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
        frame = cv2.resize(frame, (self.state_size[0], self.state_size[1]), interpolation=cv2.INTER_AREA)
        monocrome_state = frame[:, :]
        #print("monocrome shape: ", monocrome_state.shape)
        self.state_buffer.append(monocrome_state)

    def get_state_buffer(self):
        state_buffer = np.stack(self.state_buffer)
        state_buffer = np.transpose(state_buffer, (1, 2, 0))
        return state_buffer

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done, time in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target

        self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def replay2(self, batch_size):
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
        self.model.fit(states, targets_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


def test():
    env = gym.make("Breakout-v0")
    state_size = env.observation_space.shape
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    agent.load("./est-breakout.h5")
    agent.epsilon = 0.01

    frame = env.reset()
    agent.add_frame_to_buffer(frame)
    state = agent.get_state_buffer()
    for time in range(1000):
        env.render()
        action = agent.act(state)
        next_frame, reward, done, _ = env.step(action)
        agent.add_frame_to_buffer(next_frame)
        next_state = agent.get_state_buffer()
        state = next_state
        if done:
            x = state[:,:,0]
            plt.imshow(x)
            plt.show()
            print("score: {}".format(time))
            break


def train():
    env = gym.make('Breakout-v0')
    state_size = env.observation_space.shape
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    # agent.load("./est-breakout.h5")
    # agent.epsilon = 0.01
    batch_size = 64
    score_window = deque(maxlen=100)

    for e in range(EPISODES):
        frame = env.reset()
        agent.add_frame_to_buffer(frame)
        state = agent.get_state_buffer()
        score = 0
        for time in range(1000):
            action = agent.act(state)
            next_frame, reward, done, _ = env.step(action)
            agent.add_frame_to_buffer(next_frame)
            next_state = agent.get_state_buffer()

            score += reward
            # if time < 499 and done:
            #     reward = -20
            # if time == 499:
            #     reward = 20  # 1/(1 - gamma)
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
            agent.replay2(batch_size)
        if e % 100 == 0:
            agent.save("./est-breakout.h5")


if __name__ == "__main__":
    #train()
    test()

