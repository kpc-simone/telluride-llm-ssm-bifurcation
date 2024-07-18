import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import seaborn as sns

from ChasingBearEnv import *
from helper_functions_chasing_bear_env import *

'''
Script to inplement DQN for the ChasingBearEnv environment
'''

## set the device to mps or cuda if available
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

print(f"Using device: {device}")

class DQN(nn.Module):
    '''
    DQN network: Convnet followed by a fully connected layer
    '''
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_shape, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )
    #     self.conv = nn.Sequential(
    #         nn.Conv2d(input_shape, 32, kernel_size=8, stride=4),
    #         nn.ReLU(),
    #         nn.Conv2d(32, 64, kernel_size=4, stride=2),
    #         nn.ReLU(),
    #         nn.Conv2d(64, 64, kernel_size=3, stride=1),
    #         nn.ReLU()
    #     )
    #     conv_out_size = self._get_conv_out(input_shape)
    #     self.fc = nn.Sequential(
    #         nn.Linear(conv_out_size, 512),
    #         nn.ReLU(),
    #         nn.Linear(512, n_actions)
    #     )

    # def _get_conv_out(self, shape):
    #     o = self.conv(torch.zeros(1, shape))
    #     return int(torch.prod(torch.tensor(o.size())))

    # def forward(self, x):
    #     conv_out = self.conv(x).view(x.size()[0], -1)
    #     return self.fc(conv_out)
    def forward(self, x):
        return self.fc(x)
    

class DQNAgent:
    '''
    DQN Agent class. Implements the DQN algorithm.
    1. remember: Store the state, action, reward, next_state, done tuple in the memory
    2. act: Choose an action based on the epsilon-greedy policy
    3. train: Train the DQN model
    4. update_target: Update the target model
    '''
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=10000) ## this can be replaced using a SSM
        self.gamma = 0.99 ## discount factor
        self.epsilon = 1.0 ## exploration rate
        self.epsilon_min = 0.01 ## minimum exploration rate
        self.epsilon_decay = 1 - 1e-3 ## decay rate for exploration
        self.model = DQN(state_dim, action_dim).to(device)
        self.target_model = DQN(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)

    def remember(self, state, action, reward, next_state, done):
        '''
        Store the state, action, reward, next_state, done tuple in the memory
        '''
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if random.random() < self.epsilon: ## explore
            return random.choice(range(self.action_dim))
        
        else: ## exploit
            state = torch.FloatTensor(state).unsqueeze(0).to(device) ## add a batch dimension
            q_values = self.model(state) ## get the q_values
            return torch.argmax(q_values).item() ## return the action with the highest q_value
        
    def train(self, batch_size):
        '''
        Train the DQN model. 
        1. Sample a batch from the memory
        2. Compute the loss
        3. Backpropagate the loss
        4. Update the target model
        '''
        if len(self.memory) < batch_size: ## if the memory is less than the batch_size
            return
        
        batch = random.sample(self.memory, batch_size) ## sample a batch from the memory
        for state, action, reward, next_state, done in batch:
            target = reward ## initialize the target
            if not done:
                next_state = torch.FloatTensor(next_state).unsqueeze(0).to(device) ## extract the next state
                target = reward + self.gamma * torch.max(self.target_model(next_state).detach()) ## compute the target

            state = torch.FloatTensor(state).unsqueeze(0).to(device) ## extract the state
            target_f = self.model(state) ## state-action value
            target_f[0][action] = target ## update the target

            ## optimize the model
            self.optimizer.zero_grad()
            loss = nn.MSELoss()(self.model(state), target_f)
            loss.backward() ## backpropagate the loss
            self.optimizer.step() ## update the weights

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        '''
        Update the target model.
        1. Load the weights of the model into the target model

        '''
        self.target_model.load_state_dict(self.model.state_dict())


## TODO: implement the training
# env = ChasingBearEnv()
# state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
# action_dim = env.action_space.n

# agent = DQNAgent(state_dim, action_dim)
# episodes = 100
# batch_size = 32
# train_rewards = []

# for e in range(episodes):
#     state, _ = env.reset()
#     state = state.flatten()
#     total_reward = 0
#     for time in range(env.max_steps):
#         action = agent.act(state)
#         next_state, reward, done, _, _ = env.step(action)
#         next_state = next_state.flatten()
#         agent.remember(state, action, reward, next_state, done)
#         state = next_state
#         total_reward += reward
#         if done:
#             agent.update_target_model()
#             break
#         agent.train(batch_size)
#     train_rewards.append(total_reward)
#     print(f"Episode {e+1}/{episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon}")


# ## save the agent and the model
# torch.save(agent.model.state_dict(), 'dqn_model_chasing_bear.pth')
# torch.save(agent.target_model.state_dict(), 'dqn_target_model_chasing_bear.pth')

# # Plot training progress
# plt.plot(train_rewards)
# plt.xlabel('Episode')
# plt.ylabel('Total Reward')
# plt.title('Training Progress')
# sns.despine()
# plt.show()





        
        

    

