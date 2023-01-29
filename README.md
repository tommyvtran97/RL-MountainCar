# Application of Reinforcement Learning on OpenAI GYM Environment

This repository contains the implementation of several RL algorithms applied to OpenAI Gym environment. We provide several agents that are trained with different algorithms to execute a task:

- Q-learning
- SARSA (State Action Reward State Action)
- DQN (Deep Q-Network)

## Train the agent
The agent can be trained with three different RL algorithms QL, SARSA and DQN. The default algorithm is QL.

```
python mountaincar.py --train --agent QL
```

## Evaluate the agent
In a similar way the agent can be evaluated:
```
python mountaincar.py --evaluate --agent QL
```

![alt text](https://github.com/tommyvtran97/RL_DQN/blob/master/samples/DQN.gif) 




