import yaml
import gym
import argparse
import numpy as np

from utils import AlgorithmRL

# Initialize parse arguments
parser = argparse.ArgumentParser(description='Application to run the camera for data collection offline')

# Optional argument
parser.add_argument('--train', action='store_true')
parser.add_argument('--evaluate', action='store_true')

args = parser.parse_args()

if args.train:
    # Create GYM environment
    env = gym.make("MountainCar-v0")

    # Extract information from YAML
    with open("config/config.yaml", encoding="utf8") as yamlconf:
        config = yaml.safe_load(yamlconf)

    # Initialize learning parameters
    learning_rate = config['mountaincar']['learning_rate']
    discount = config['mountaincar']['discount']
    episodes = config['mountaincar']['episodes']

    # Discrete observation space
    discrete_os_size = [20] * len(env.observation_space.high)
    discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / discrete_os_size

    # Exploration settings
    epsilon = config['mountaincar']['epsilon'] 
    start_epsilon_decay = config['mountaincar']['start_epsilon_decay']
    end_epsilon_decay = episodes / 2
    epsilon_decay_value = epsilon/(end_epsilon_decay - start_epsilon_decay)

    q_table = np.random.uniform(low=-2, high=0, size=(discrete_os_size + [env.action_space.n]))

    # Initilize RL object
    RL = AlgorithmRL(env.observation_space.low, discrete_os_win_size)

    # Initialize empty array to store rewards per episode
    total_score = np.zeros(episodes)

    for episode in range(episodes):
        env.reset()
        discrete_state = RL.get_discrete_state(np.array([0, 0]))
        end_training = False

        reward_score = 0
        while not end_training:

            if np.random.random() > epsilon:
                action = np.argmax(q_table[discrete_state])
            else:
                action = np.random.randint(0, env.action_space.n)

            new_state, reward, end_training, truncated, info = env.step(action)

            # Perform Q-learning
            discrete_state = RL.Qlearning(action, reward, new_state, discrete_state, q_table, env.goal_position, learning_rate, discount, end_training)

            # Accumulate reward
            reward_score += reward

        total_score[episode] = reward_score
        print(f'Epsiode: {episode} - Reward: {total_score[episode]}')

        if end_epsilon_decay >= episode >= start_epsilon_decay:
            epsilon -= epsilon_decay_value

    # Save optimized Q-table policy
    np.save("models/qtable_mountain.npy", q_table)

    env.close()

if args.evaluate:
    # Create GYM environment
    agent = gym.make("MountainCar-v0", render_mode='human')
    agent.reset()

    # Discrete observation space
    discrete_os_size = [20] * len(agent.observation_space.high)
    discrete_os_win_size = (agent.observation_space.high - agent.observation_space.low) / discrete_os_size

    # Initilize RL object
    RL = AlgorithmRL(agent.observation_space.low, discrete_os_win_size)

    # Evaluate agent
    RL.evaluate_agent(agent, "models/qtable_mountain.npy")