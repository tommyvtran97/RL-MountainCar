import numpy as np
import torch
import random

from collections import namedtuple, deque

class AlgorithmRL():
    def __init__(self, observation_low, discrete_os_win_size) -> None:
        self.observation_low = observation_low
        self.discrete_os_win_size = discrete_os_win_size

    def get_discrete_state(self, state):
        discrete_state = (state - self.observation_low) / self.discrete_os_win_size
        return tuple(discrete_state.astype(np.int32))

    def Qlearning(self, env, state, q_table, learning_rate, discount, epsilon):

        discrete_state = self.get_discrete_state(state)
        end_training = False

        reward_score = 0
        while not end_training:
            if np.random.random() > epsilon:
                action = np.argmax(q_table[discrete_state])
            else:
                action = np.random.randint(0, env.action_space.n)

            new_state, reward, end_training, _, _ = env.step(action)

            new_discrete_state = self.get_discrete_state(new_state)

            if not end_training:
                max_future_q = np.max(q_table[new_discrete_state])
                current_q = q_table[discrete_state + (action, )]
                new_q = (1 - learning_rate) * current_q + learning_rate * (reward + discount * max_future_q)
                q_table[discrete_state + (action, )] = new_q

            elif new_state[0] >= env.goal_position:
                q_table[discrete_state + (action, )] = 0

            discrete_state = new_discrete_state

            reward_score += reward

        return q_table, reward_score

    def SARSA(self, env, state, q_table, learning_rate, discount, epsilon):

        discrete_state = self.get_discrete_state(state)
        end_training = False

        reward_score = 0
        while not end_training:
            if np.random.random() > epsilon:
                action = np.argmax(q_table[discrete_state])
            else:
                action = np.random.randint(0, env.action_space.n)

            new_state, reward, end_training, _, _ = env.step(action)

            new_discrete_state = self.get_discrete_state(new_state)

            if np.random.random() > epsilon:
                action_ = np.argmax(q_table[new_discrete_state])
            else:
                action_ = np.random.randint(0, env.action_space.n)

            if not end_training:
                future_q = q_table[new_discrete_state + (action_, )]
                current_q = q_table[discrete_state + (action, )]
                new_q = (1 - learning_rate) * current_q + learning_rate * (reward + discount * future_q)
                q_table[discrete_state + (action, )] = new_q

            elif new_state[0] >= env.goal_position:
                q_table[discrete_state + (action, )] = 0

            discrete_state = new_discrete_state

            reward_score += reward

        return q_table, reward_score

    def DQN(self, env, policy_net, target_net, memory, state, batch_size, learning_rate, discount, epsilon, device, update_rate):

        optimizer = torch.optim.AdamW(policy_net.parameters(), lr=learning_rate, amsgrad=True)
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        end_training = False

        reward_score = 0
        iteration = 0
        while not end_training:

            if np.random.random() > epsilon:
                with torch.no_grad():
                   action = policy_net(state).max(1)[1][0].cpu().numpy()
            else:
                action = np.random.randint(0, env.action_space.n)

            state_, reward, end_training, truncated, info = env.step(action)

            if end_training:
                state_ = None
            else:
                state_ = torch.tensor(state_, dtype=torch.float32, device=device).unsqueeze(0)
        
            # Tensor conversion
            action = torch.tensor(action, device=device).unsqueeze(0)
            reward = torch.tensor([reward], device=device)

            # Append to experience buffer
            memory.push(state, action, state_, reward)

            state = state_

            # Model optimization
            if len(memory) > batch_size:
                transition_state = memory.sample(batch_size)

                batch = memory.transition(*zip(*transition_state))

                non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                    batch.next_state)), device=device, dtype=torch.bool)
                non_final_next_states = torch.cat([s for s in batch.next_state
                                                            if s is not None])

                state_batch = torch.cat(batch.state)
                action_batch = torch.cat(batch.action)
                reward_batch = torch.cat(batch.reward)

                state_action_values = policy_net(state_batch).gather(1, action_batch.unsqueeze(-1))
  
                next_state_values = torch.zeros(batch_size, device=device)
                with torch.no_grad():
                    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
                expected_state_action_values = (next_state_values * discount) + reward_batch

                # Compute Huber loss
                criterion = torch.nn.SmoothL1Loss()
                loss = criterion(state_action_values, expected_state_action_values.unsqueeze(-1))
                
                # Optimize the model
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

                if iteration % update_rate:
                    # Update target network     
                    target_net_state_dict = target_net.state_dict()
                    policy_net_state_dict = policy_net.state_dict()
                    for key in policy_net_state_dict:
                        target_net_state_dict[key] = policy_net_state_dict[key]*0.005 + target_net_state_dict[key]*(1-0.005)
                    target_net.load_state_dict(target_net_state_dict)

            iteration += 1
            reward_score += reward

        return policy_net, reward_score

    def evaluate_agent(self, device, state, agent, env, q_table=None, policy_net=None):
        
        if agent in ['QL', 'SARSA']:
            discrete_state = self.get_discrete_state(np.array([0, 0]))
            model = np.load(q_table)

            reward_total = 0
            while True:
                action = np.argmax(model[discrete_state])
                new_state, reward, done, _, _ = env.step(action)
                discrete_state = self.get_discrete_state(new_state)

                reward_total += reward

                if done:
                    print(f'Total Reward of {agent}: {reward_total}')
                    break

        if agent == 'DQN':
            model = FCNet(env.observation_space.shape[0], env.action_space.n).to(device)
            model.load_state_dict(torch.load(policy_net, map_location=device))
            model.eval()

            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            
            reward_total = 0
            while True:
                action = model(state).max(1)[1][0].cpu().numpy()
                state, reward, done, _, _ = env.step(action)
                state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

                reward_total += reward

                if done:
                    print(f'Total Reward of {agent}: {reward_total}')
                    break


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

        self.transition = namedtuple('Transition',
                                ('state', 'action', 'next_state', 'reward'))
    def push(self, *args):
        """Save a transition"""
        self.memory.append(self.transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class FCNet(torch.nn.Module):

    def __init__(self, n_observations, n_actions):
        super(FCNet, self).__init__()

        self.layer1 = torch.nn.Linear(n_observations, 128)
        self.layer2 = torch.nn.Linear(128, 128)
        self.layer3 = torch.nn.Linear(128, n_actions)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = torch.nn.functional.relu(self.layer1(x))
        x = torch.nn.functional.relu(self.layer2(x))
        x = self.layer3(x)

        return x
