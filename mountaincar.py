import numpy as np
import yaml
import gym
import argparse
import torch

from utils import AlgorithmRL, FCNet, ReplayMemory

# Initialize parse arguments
parser = argparse.ArgumentParser(description='Application to run RL algorithms')

# Optional argument
parser.add_argument('--agent', type=str, default='QL')
parser.add_argument('--train', action='store_true')
parser.add_argument('--evaluate', action='store_true')

args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Extract information from YAML
with open("config/config.yaml", encoding="utf8") as yamlconf:
    config = yaml.safe_load(yamlconf)

if args.train:
    # Create GYM environment
    env = gym.make("MountainCar-v0")

    # Initialize learning parameters
    if args.agent == 'DQN':
        batch_size = config['mountaincar'][args.agent]['batch_size']
        update_rate = config['mountaincar'][args.agent]['update_rate']
    if args.agent in ['QL', 'SARSA', 'DQN']:    
        learning_rate = config['mountaincar'][args.agent]['learning_rate']
        discount = config['mountaincar'][args.agent]['discount']
        episodes = config['mountaincar'][args.agent]['episodes']

    # Discrete observation space
    discrete_os_size = [20] * len(env.observation_space.high)
    discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / discrete_os_size

    # Exploration settings
    epsilon = config['mountaincar'][args.agent]['epsilon'] 
    start_epsilon_decay = config['mountaincar'][args.agent]['start_epsilon_decay']
    end_epsilon_decay = episodes / 2
    epsilon_decay_value = epsilon/(end_epsilon_decay - start_epsilon_decay)
    
    # Initilize Q-table
    if args.agent in ['QL', 'SARSA']:
        q_table = np.random.uniform(low=-2, high=0, size=(discrete_os_size + [env.action_space.n]))

    # Initilize RL object
    RL = AlgorithmRL(env.observation_space.low, discrete_os_win_size)

    # Initialize empty array to store rewards per episode
    total_score = np.zeros(episodes)
    
    # Initilize experience buffer
    memory = ReplayMemory(10000)

    if args.agent == 'DQN':
        policy_net = FCNet(env.observation_space.shape[0], env.action_space.n).to(device) 
        target_net = FCNet(env.observation_space.shape[0], env.action_space.n).to(device) 

    top_agent = -10000
    for episode in range(episodes):
        state, _ = env.reset()
        
        if args.agent == 'QL':
            q_table, reward_score = RL.Qlearning(env, state, q_table, learning_rate, discount, epsilon)
        if args.agent == 'SARSA':
            q_table, reward_score = RL.SARSA(env, state, q_table, learning_rate, discount, epsilon)
        if args.agent == 'DQN':
            policy_net, reward_score = RL.DQN(env, policy_net, target_net, memory, state, batch_size, learning_rate, discount, epsilon, device, update_rate)

        if end_epsilon_decay >= episode >= start_epsilon_decay:
            epsilon -= epsilon_decay_value

        total_score[episode] = reward_score
        print(f'Epsiode: {episode} - Reward: {total_score[episode]}')

        if total_score[episode] > top_agent:
            # Save optimized Q-table policy
            if args.agent in ['QL', 'SARSA']:
                np.save(f'{args.agent}_agent.npy', q_table)
            # Save DQN model
            if args.agent == 'DQN':
                torch.save(policy_net.state_dict(), "DQN_agent.pt")

            # Update agent score
            top_agent = reward_score

    env.close()

if args.evaluate:

    # Create GYM environment
    env = gym.make("MountainCar-v0", render_mode='human')
    state, _ = env.reset()

    # Discrete observation space
    discrete_os_size = [20] * len(env.observation_space.high)
    discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / discrete_os_size

    # Initilize RL object
    RL = AlgorithmRL(env.observation_space.low, discrete_os_win_size)
    model = config['mountaincar'][args.agent]['model']

    # Evaluate agent
    if args.agent in ['QL', 'SARSA']:
        RL.evaluate_agent(device, state, args.agent, env, q_table=model) 
    if args.agent == 'DQN':
        RL.evaluate_agent(device, state, args.agent, env, policy_net=model)