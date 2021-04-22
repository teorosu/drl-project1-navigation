# Project 1: Navigation

## Algorithm

The final state of the solution contains the following algorithms:

- Deep Q-Network
- Prioritized Experience Learning
- Dueling Network Architectures for Deep RL

The agent logic can be split in:
1. Act/Sample. The agent interacts with the environment. Each experience is stored in an agent memory component.

2. Learn. The agent will use retrieve batches of experience from the memory component in order to learn.

### Worklog / Details

- I started initially with a DQN network (without dueling) and a non-prioritized experience memory model. The agent learned in around 500 episodes.

- I replaced the Experience Memory with a Prioritized Experience Memory. This change was not a big improvement ( to my disapointment ) and the agent was learning in around 400-500 episodes.

- My next step was to implement the Dueling Model DQN. This feature decreased the time needed for the agent to learn, getting to ~250 episodes to learn.

## Hyperparameters

The main hyperparameters:

* *BUFFER_SIZE* - the agent (experience) memory size measured in entries. The value I used is 10^5. I tried higher values, but the agent was much slower and I did not see relevant improvements.

* *BATCH_SIZE* - the size of a batch extracted from "Memory". I used a size of 64. I did not experiment with many values.

* *GAMMA* - the discount factor, is the discount applied to future events. For this param 0.99 was used.

* *TAU* - the local network parameter contribution ( fraction ) to the target network parameters (set to 0.01)

* *LR* - the learning rate of the networks ( set to 0.0005 )

* *UPDATE_EVERY* - the number of agent steps before a learning "session" (set to 4)

* *OPTIMIZER_LR* - (gamma - multiplicative factor) learning rate decay  ( set to 0.9999) 

* *nonzero_offset* - in context of Prioritized Experience Replay, this is the value added to all errors so that no error is zero and each experience has a chance to be picked

## Model Architecture

### Baseline model

I started with a Deep Q-Network using Experience Replay. This is/was my baseline model. I tried to play with 
hidden layers and number of nodes. Some experiments showed that more than 3 layer did not improve the
agent. More than this, having a lot of nodes per layer ( > 512 ) was not an improvement. I settled to 2 layers, each one 64 nodes.

### Prioritized learning

For some context, my personal belief was that Prioritized Experience Learning would be impactful. So I replaced the Experience Learning component with a Prioritized Experience Learning implementation. Unfortunately this did not significantly reduce the number of episodes needed to learn. The agent still needed around 400 episodes. 

### Dueling networks

The next idea to improve the agent was the dueling network. I changed the model architecture according to the dueling netowrk paper. In this model, I tried tweaking the number of layers and nodes in layers and ended up with a network 

```
                                    |- value (1 node) ----------------|
(Input) - (128 nodes) - (64 nodes) -|                                 |=> result
                                    |- advantage (37 nodes)-----------|
```

### Learning plot

![](learning_graph.png)

## Conclusion / Future work

While the steps above improve the agent, a more programatic and organised approach to tweaking hyperparameters and the model would be useful. Most of the tweaking was manual tweaking. 