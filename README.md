# CS4100 Knights Archers Zombies (KAZ) Centralized and Decentralized Training
Knights Archers Zombies is a game provided by the ButterFly enviroment in Petting Zoo. The game consists of archers and knights fighting off zombies that are coming in waves. Petting Zoo in general offers a variety of multi-agent enviroments to train and test different Reinforcement Learning algorithms. Our goal was to implement Decentralized and Centralized training to the archers and knights in hope to see optimal performance. For Decentralized Training, we implemented Deep Q Networks where each agent updates its own policy without considering the behavior of the other agents in the enviroment. On the other hand, Centralized Training was 


## Centralized Training

```bash
pip install gym numpy torch pygame matplotlib tianshou pettingzoo
```

##### Training
```bash
python KAZ_centralize_train.py
```

##### Visulizatoin
```bash
python KAZ_centralize_train.py --observe_only
```

##### Data Viewing
```bash
tensorboard --logdir=data
```
