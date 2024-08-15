# Knights Archers Zombies (KAZ) Centralized Training

This repository contains code for training reinforcement learning (RL) agents to play the "Knights Archers Zombies" game using centralized training with the Tianshou library and PettingZoo environment.

### Requirements

To set up the environment and run the code, you need the following Python packages:

```bash
pip install gym numpy torch pygame matplotlib tianshou pettingzoo
```

## Running the Code

### Training the Agents

To start training the agents, use:

```bash
python KAZ_centralize_train.py
```

This command will run the training process with default parameters.

### Observing the Agents

If you want to observe a trained policy without training, use:

```bash
python KAZ_centralize_train.py --observe_only
```

This will render a single environment so you can watch the trained agents in action.

## Command-Line Arguments

Customize the training process using the following arguments:

- `--seed`: Random seed for reproducibility (default: 1626)
- `--test_eps`: Epsilon for exploration during testing (default: 0.05)
- `--train_eps`: Epsilon for exploration during training (default: 0.7)
- `--buffer_cap`: Capacity of the replay buffer (default: 50000)
- `--lr`: Learning rate for the optimizer (default: 0.0003)
- `--gamma`: Discount factor (default: 0.9)
- `--num_archers`: Number of archer agents (default: 2)
- `--num_knights`: Number of knight agents (default: 2)
- `--epochs`: Number of training epochs (default: 2)
- `--steps_per_epoch`: Number of steps per epoch (default: 2000)
- `--batch_size`: Batch size for training (default: 64)
- `--hidden_layers`: List of hidden layer sizes for the neural network (default: [512, 256])
- `--num_envs`: Number of environments for training (default: 10)
- `--render_freq`: Frequency of rendering during observation (default: 0.005)
- `--win_rate`: Desired win rate for the training (default: 0.6)
- `--observe_only`: Run in observation mode only (default: False)
- `--agent_id`: Player ID (1 or 2, default: 2)
- `--resume_path`: Path to resume training from a checkpoint (default: '')
- `--opponent_path`: Path to an opponent policy (default: '')
- `--device`: Device for training (default: 'cuda' if available, otherwise 'cpu')

## Logging and Results

Training results and logs are saved in the `data` directory:

- Training rewards and losses are plotted and saved as `rewards_plot.png` and `losses_plot.png`.
- TensorBoard logs can be viewed with:

```bash
tensorboard --logdir=data
```
