import gymnasium as gym
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


env = gym.make('CartPole-v1')
# For newer gym versions, reset might return obs, info.
# We'll assume it returns just obs based on the original code's usage.
# If it causes issues, use: observation, _ = env.reset()
observation = env.reset()
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
        # Reset environment for each game
        current_observation = env.reset()
        if isinstance(current_observation, tuple): # Handle gym's new API
            current_observation = current_observation[0]
        
        for step_index in range(goal_steps):
            action = random.randrange(0, 2)
            observation, reward, done, truncated, info = env.step(action) # gym 0.26+
            # For older gym: observation, reward, done, info = env.step(action)
            # done = done or truncated # for gym 0.26+

            if len(previous_observation) > 0:
                game_memory.append([previous_observation, action])

            previous_observation = observation
            score += reward
            if done or truncated: # gym 0.26+
                break
        
        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                if data[1] == 1:
                    output = [0, 1]
                elif data[1] == 0:
                    output = [1, 0]
                training_data.append([data[0], output])
        
        # env.reset() # Already reset at the start of the game loop
    
    # print(len(accepted_scores))
    # print(accepted_scores)

    return training_data


class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 52)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(52, output_size)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x


def train_model(training_data, model=None, epochs=10):
    X_list = [i[0] for i in training_data]
    y_list = [i[1] for i in training_data]

    X = torch.tensor(np.array(X_list), dtype=torch.float32)
    y = torch.tensor(np.array(y_list), dtype=torch.float32)
    
    print("X shape: ", X.shape)
    print("y shape: ", y.shape)

    if model is None:
        model = Net(input_size=X.shape[1], output_size=y.shape[1])

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        # if (epoch + 1) % 1 == 0: # Or some other interval
        #     print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
            
    return model


def obs2input(obs):
    return torch.tensor(obs, dtype=torch.float32).unsqueeze(0)


def out2action(out_tensor):
    return torch.argmax(out_tensor).item()


training_data = model_data_preparation()
trained_model = train_model(training_data)


scores = []
lasts_pos = []
match_data = []
for each_game in range(100):
    game_data = []
    score = 0
    observation = env.reset()
    if isinstance(observation, tuple): # Handle gym's new API
        observation = observation[0]

    for step_index in range(goal_steps):
        # env.render()
        X_tensor = obs2input(observation)
        
        trained_model.eval()
        with torch.no_grad():
            y_pred_tensor = trained_model(X_tensor)
        
        action = out2action(y_pred_tensor)

        out_one_hot = [1.0, 0.0] # Use float for potential tensor conversion
        if action == 1:
            out_one_hot = [0.0, 1.0]

        game_data.append([observation, out_one_hot]) # Storing one-hot action

        # observation, reward, done, info = env.step(action) # older gym
        observation, reward, done, truncated, info = env.step(action) # gym 0.26+
        # done = done or truncated # for gym 0.26+

        score += reward
        if done or truncated: # gym 0.26+
            break
    
    # Storing final observation's x-position
    # Ensure observation is a numpy array before indexing if it's not already
    obs_np = observation if isinstance(observation, np.ndarray) else np.array(observation)
    match_data.append([score, obs_np[0], game_data])
    lasts_pos.append(obs_np[0])
    # env.reset() # Reset is handled at the start of the game loop
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
    if isinstance(observation, tuple): # Handle gym's new API
        observation = observation[0]

    for step_index in range(goal_steps):
        # env.render()
        X_tensor = obs2input(observation)

        trained_model.eval()
        with torch.no_grad():
            y_pred_tensor = trained_model(X_tensor)
            
        action = out2action(y_pred_tensor)

        # Storing raw model output (as numpy array)
        game_data.append([observation, y_pred_tensor.squeeze(0).detach().numpy()])

        observation, reward, done, truncated, info = env.step(action)

        score += reward
        if done or truncated: # gym 0.26+
            break
    
    obs_np = observation if isinstance(observation, np.ndarray) else np.array(observation)
    match_data.append([score, obs_np[0], game_data])
    lasts_pos.append(obs_np[0])
    # env.reset() # Reset is handled at the start of the game loop
    scores.append(score)

print(scores)
print(lasts_pos)
print('Average Score:', sum(scores) / len(scores))
print('Pos variance:', np.var(lasts_pos))

env.close()
