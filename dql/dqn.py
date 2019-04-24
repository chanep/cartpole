# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Activation
from keras.optimizers import Adam
from keras.regularizers import l2

EPISODES = 10000

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=5000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.learning_rate = 0.0002
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        reg = None
        # reg = l2(0.0001)

        model = Sequential()
        model.add(Dense(48, input_dim=self.state_size, activation='relu', kernel_regularizer=reg))
        # model.add(Dense(96, kernel_regularizer=reg))
        # model.add(Activation('relu'))
        # model.add(Dense(48, kernel_regularizer=reg))
        # model.add(BatchNormalization())
        # model.add(Activation('relu'))
        model.add(Dense(48, activation='relu', kernel_regularizer=reg))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
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
        for state, action, reward, next_state, done in minibatch:
            next_states.append(next_state[0])
            states.append(state[0])

        next_states = np.stack(next_states, axis=0)
        states = np.stack(states, axis=0)
        Qn = self.model.predict(next_states)
        Q = self.model.predict(states)

        targets_f = []
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(Qn[i]))
            target_f = Q[i]
            target_f[action] = target
            targets_f.append(target_f)

        targets_f = np.stack(targets_f, axis=0)
        self.model.fit(states, targets_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    # agent.load("./est-dqn.h5")
    # agent.epsilon = 0.01
    done = False
    batch_size = 64
    score_window = deque(maxlen=100)

    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in range(1000):
            # env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            if time < 499 and done:
                reward = -50
            if time == 499:
                reward = 10
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                score_window.append(time)
                score_avg = sum(score_window) / len(score_window)
                print("episode: {}/{}, score: {}, avg: {:.1f}, e: {:.2}"
                      .format(e, EPISODES, time, score_avg, agent.epsilon))
                break
        if len(agent.memory) > batch_size:
            agent.replay2(batch_size)
        if e % 100 == 0:
            agent.save("./est-dqn.h5")