# CS4100 Knights Archers Zombies (KAZ) Centralized and Decentralized Training
Knights Archers Zombies is a game provided by the ButterFly enviroment in Petting Zoo. The game consists of archers and knights fighting off zombies that are coming in waves. Petting Zoo in general offers a variety of multi-agent enviroments to train and test different Reinforcement Learning algorithms. Our goal was to implement Decentralized and Centralized training to the archers and knights in hope to see optimal performance. For Decentralized Training, we implemented Deep Q Networks where each agent updates its own policy without considering the behavior of the other agents in the enviroment. On the other hand, Centralized Training was 
## Decentralized Training
### Installation
First, install the necessary dependencies:
```bash
pip install gym numpy torch pygame matplotlib tianshou pettingzoo
```

### Training
To start training in a decentralized setting, run the following command:

```bash
python KAZ_centralize_train.py
```

### Visualization
To observe the agents' performance during Centralized training:

```bash
python KAZ_centralize_train.py --observe_only
```

### Data Viewing
You can monitor the training progress and metrics using TensorBoard:

```bash
tensorboard --logdir=data
```

### Notes
- **Visualization**: The `--observe_only` flag lets you watch the trained agents in action without further training, providing insight into their learned behaviors.
- **TensorBoard**: Use TensorBoard for a detailed view of training metrics, such as loss curves and reward progress, to evaluate the effectiveness of centralized training.
