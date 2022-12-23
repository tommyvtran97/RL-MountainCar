import numpy as np

class AlgorithmRL():
    def __init__(self, observation_low, discrete_os_win_size) -> None:
        self.observation_low = observation_low
        self.discrete_os_win_size = discrete_os_win_size

    def get_discrete_state(self, state):
        discrete_state = (state - self.observation_low) / self.discrete_os_win_size
        return tuple(discrete_state.astype(np.int32))

    def Qlearning(self, action, reward, new_state, old_discrete_state, q_table, goal_position, learning_rate, discount, end_training):

        new_discrete_state = self.get_discrete_state(new_state)

        if not end_training:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[old_discrete_state + (action,)]
            new_q = (1 - learning_rate) * current_q + learning_rate * (reward + discount * max_future_q)
            q_table[old_discrete_state + (action,)] = new_q

        elif new_state[0] >= goal_position:
            q_table[old_discrete_state + (action,)] = 0

        return new_discrete_state

    def evaluate_agent(self, agent, q_table):

        discrete_state = self.get_discrete_state(np.array([0, 0]))
        q_table = np.load(q_table)

        while True:
            action = np.argmax(q_table[discrete_state])
            new_state, _, done, _, _ = agent.step(action)
            discrete_state = self.get_discrete_state(new_state)

            if done:
                break