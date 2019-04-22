
import gym
import random
import numpy as np
from keras.models     import Sequential
from keras.layers     import Dense
from keras.optimizers import Adam


env = gym.make('CartPole-v1')
env.reset()
goal_steps = 500
score_requirement = 60
intial_games = 1000


def model_data_preparation():
    training_data = []
    accepted_scores = []
    for game_index in range(intial_games):
        score = 0
        game_memory = []
        previous_observation = []
        for step_index in range(goal_steps):
            action = random.randrange(0, 2)
            observation, reward, done, info = env.step(action)

            if len(previous_observation) > 0:
                game_memory.append([previous_observation, action])

            previous_observation = observation
            score += reward
            if done:
                break

        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                if data[1] == 1:
                    output = [0, 1]
                elif data[1] == 0:
                    output = [1, 0]
                training_data.append([data[0], output])

        env.reset()

    # print(len(accepted_scores))
    # print(accepted_scores)

    return training_data


def build_model(input_size, output_size):
    model = Sequential()
    model.add(Dense(128, input_dim=input_size, activation='relu'))
    model.add(Dense(52, activation='relu'))
    model.add(Dense(output_size, activation='linear'))
    model.compile(loss='mse', optimizer=Adam())
    return model


def train_model(training_data, model=None, epochs=10):

    X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]))
    print("X: ", X.shape)
    y = np.array([i[1] for i in training_data]).reshape(-1, len(training_data[0][1]))
    print("y: ", y.shape)

    if model is None:
        model = build_model(input_size=len(X[0]), output_size=len(y[0]))

    model.fit(X, y, epochs=epochs, verbose=0)
    return model


def obs2input(obs):
    return np.array([obs])


def out2action(out):
    return np.argmax(out)


training_data = model_data_preparation()
trained_model = train_model(training_data)


scores = []
lasts_pos = []
match_data = []
for each_game in range(100):
    game_data = []
    score = 0
    observation = env.reset()
    for step_index in range(goal_steps):
        # env.render()
        X = obs2input(observation)
        y = trained_model.predict_on_batch(X)
        action = out2action(y)

        out = [1, 0]
        if action == 1:
            out = [0, 1]

        game_data.append([observation, out])

        observation, reward, done, info = env.step(action)

        score += reward
        if done:
            break

    match_data.append([score, observation[0], game_data])

    lasts_pos.append(observation[0])
    env.reset()
    scores.append(score)

print(scores)
print(lasts_pos)
print('Average Score:', sum(scores) / len(scores))
print('Pos variance:', np.var(lasts_pos))



match_data_sorted = sorted(match_data, key=lambda m: (m[0] + ((2.4 - abs(m[1])) / 2.4)), reverse=True)
training_data = []
for md in match_data_sorted[:50]:
    for gd in md[2]:
        training_data.append([gd[0], gd[1]])


trained_model = train_model(training_data, model=trained_model, epochs=30)

scores = []
lasts_pos = []
match_data = []
for each_game in range(100):
    game_data = []
    score = 0
    observation = env.reset()
    for step_index in range(goal_steps):
        # env.render()
        X = obs2input(observation)
        y = trained_model.predict_on_batch(X)
        action = out2action(y)

        game_data.append([observation, y])

        observation, reward, done, info = env.step(action)

        score += reward
        if done:
            break

    match_data.append([score, observation[0], game_data])

    lasts_pos.append(observation[0])
    env.reset()
    scores.append(score)

print(scores)
print(lasts_pos)
print('Average Score:', sum(scores) / len(scores))
print('Pos variance:', np.var(lasts_pos))
