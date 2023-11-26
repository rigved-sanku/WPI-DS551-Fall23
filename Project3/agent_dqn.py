#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque, namedtuple
from itertools import count
from typing import List
import os
import sys
import wandb

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim


from agent import Agent
from dqn_model import DQN

wandb.init(project='RL3',entity='sankurigved')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))

torch.manual_seed(595)
np.random.seed(595)
random.seed(595)
#Params from the original paper
CONSTANT = 200000

# The discount factor for future rewards in the reinforcement learning setting
GAMMA = 0.99

# The size of the mini-batch used in training, set to 32.
BATCH_SIZE = 32

# The size of the replay buffer, which is a memory structure used to store and sample experiences for training.
BUFFER_SIZE = 10000

# Epsilon pararmeters for the decay
# The initial exploration rate, set to 1 (meaning 100% exploration).
EPSILON = 1
# The final exploration rate, set to 0.025.
EPSILON_END = 0.025
# The number of steps after which epsilon starts decaying.
DECAY_EPSILON_AFTER = 3000


# Updating the model params
# The frequency (in steps) at which the target Q-network is updated.
TARGET_UPDATE_FREQUENCY = 5000
# The frequency (in steps) at which the model is saved.
SAVE_MODEL_AFTER = 5000

# EPSILON_DECAY=0.5
# Learning rate for the optimizer used during training.
LEARNING_RATE = 1.5e-4

# Transition is defined with fields state, action, reward, and next_state.
# This structure is commonly used to represent a transition in a reinforcement learning setting, where an agent transitions from one state to another by taking an action and receiving a reward.
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))

# This line initializes a deque (double-ended queue) named reward_buffer with a single element 0.0 and a maximum length of 100.
# A deque is useful for efficiently maintaining a fixed-size buffer of elements, in this case, the last 100 rewards received by the agent.
reward_buffer = deque([0.0], maxlen=100)

class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize everything you need here.
        For example:
            paramters for neural network
            initialize Q net and target Q net
            parameters for repaly buffer
            parameters for q-learning; decaying epsilon-greedy
            ...
        """
        # self.EPSILON= args.EPSILON
        # self.EPSILON_END=args.EPSILON_END
        # self.EPSILON_DECAY=args.EPSILON_DECAY
        # self.DECAY_EPSILON_AFTER=args.DECAY_EPSILON_AFTER
        # self.TARGET_UPDATE_FREQUENCY=args.TARGET_UPDATE_FREQUENCY
        # self.BUFFER_SIZE=args.BUFFER_SIZE

        super(Agent_DQN,self).__init__(env)
        ###########################
        # YOUR IMPLEMENTATION HERE #

        # Sets the environment (env) for the agent. It assumes that env is an environment object (possibly from OpenAI Gym).
        self.env = env
        # Retrieves the number of possible actions from the environment's action space.
        self.action_count = self.env.action_space.n

        # Sets the number of input channels for the neural network to 4.
        # This is likely the number of frames that are stacked together as input.
        in_channels = 4  # (R, G, B, Alpha)

        # Creates an instance of the DQN class, representing the Q-network. It takes the number of input channels and the
        # action count as parameters and moves the network to the specified device (CPU or GPU).
        self.Q = DQN(in_channels, self.action_count).to(device)

        # Creates a target Q-network (Q_cap).
        self.Q_cap = DQN(in_channels, self.action_count).to(device)
        # Loads the state dictionary of the main Q-network into the target Q-network. This initializes the target Q-network with the same weights as the main Q-network.
        self.Q_cap.load_state_dict(self.Q.state_dict())
        # Creates an Adam optimizer for updating the parameters of the Q-network during training.
        # It operates on the parameters of the main Q-network.
        self.optimizer = optim.Adam(self.Q.parameters(), lr=LEARNING_RATE)

        # Initializes a deque named buffer with a maximum length of BUFFER_SIZE.
        # This deque is used for experience replay, storing the agent's experiences (state, action, reward, next_state) during training.
        self.buffer=deque([], maxlen=BUFFER_SIZE)

        self.traning=0
        if args.test_dqn:
            #you can load your model here
            print('loading trained model')

            # Loads the pre-trained model from the file 'final_dqn_model.pth' using PyTorch's torch.load function.
            test=torch.load('final_dqn_model.pth')
            # Loads the state dictionary of the pre-trained model into the main Q-network (self.Q).
            # This initializes the Q-network with the weights of the pre-trained model.
            self.Q.load_state_dict(test)
            ###########################
            # YOUR IMPLEMENTATION HERE #


    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary.
        If no parameters need to be initialized, you can leave it as blank.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #

        ###########################
        pass


    def make_action(self, observation, test=False):
        """
        Return predicted action of your agent
        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)
        Return:
            action: int
                the predicted action from trained model
        """

        # Converts the observation (a stack of 4 preprocessed frames) to a NumPy array of type float32.
        # The pixel values are scaled between 0 and 1 by dividing by 255.
        state = np.asarray(observation,dtype=np.float32)/255

        # Transposes the dimensions of the state array. It changes the shape from (84, 84, 4) to (4, 84, 84).
        state=state.transpose(2,0,1)

        # Converts the NumPy array state to a PyTorch tensor and adds an extra dimension at the beginning.
        # This is done to create a mini-batch of size 1, as neural networks usually expect input in the form of mini-batches.
        state = torch.from_numpy(state).unsqueeze(0)

        # Passes the preprocessed frames through the Q-network (self.Q) to obtain the Q-values for each action.
        Q_new = self.Q(state)

        # Finds the index of the action with the highest Q-value.
        # The [0] at the end extracts the value from the tensor, as torch.argmax returns an index tensor.
        best_q_idx = torch.argmax(Q_new,dim=1)[0]
        # This sequence of operations detaches the tensor from the computation graph (making it a standalone tensor) and converts it to a Python scalar using item().
        action_idx = best_q_idx.detach().item()

        #  finds the index of the action with the highest Q-value in the Q-values tensor and extracts it as a Python scalar.
        # This index corresponds to the predicted action based on the Q-network's output.
        return action_idx

    # The *args syntax allows the method to accept any number of arguments.
    # The -> None indicates that the method does not return any value.
    def push(self,*args)->None:
        """ You can add additional arguments as you need.
        Push new data to buffer and remove the old one if the buffer is full.

        Hints:
        -----
            you can consider deque(maxlen = 10000) list
        """
        # This constructs a Transition object using the arguments passed to the push method.
        # This line appends a new Transition object to the deque. If the deque is already at its maximum length (as specified during its creation),
        # it automatically removes the oldest element to make room for the new one.
        self.buffer.append(Transition(*args))



    def replay_buffer(self, batch_size):
        """ You can add additional arguments as you need.
        Select batch from buffer.
        """
        #  Initializes an empty list called samples to store the sampled transitions.
        samples = []

        # This loop iterates batch_size times to sample that many transitions from the replay buffer.
        for i in range(batch_size):

            # This index is used to randomly select a transition for sampling.
            idx = random.randrange(len(self.buffer))
            # Appends the selected transition at index idx to the samples list.
            samples.append(self.buffer[idx])
            # Removes the selected transition from the replay buffer to avoid re-sampling the same transition.
            del self.buffer[idx]

        return samples


    def train(self):

        """
        Implement your training algorithm here
        """

        # Initialize the episode counter.
        episode = 0

        # Initialize a variable to track the previous mean episode reward.
        exp_reward=0

        # The training loop continues as long as the mean reward in the reward_buffer is less than 50.
        while np.mean(reward_buffer) < 50:

            # Print the current episode number.
            print("Doing Episode", episode)
            # Initialize a timestamp counter for the current episode.
            t_stamp=0
            # Initialize the cumulative reward for the current episode.
            epi_reward=0
            #put counter ++ in the end of while loop
            # Reset the environment to start a new episode and get the initial state.
            curr_state = self.env.reset()

            # Enter the inner loop, which runs until the episode is done.
            while True:
                # Check if the episode number is greater than a threshold for epsilon decay threshold
                if episode > DECAY_EPSILON_AFTER:
                    # Decay the exploration parameter (epsilon) over time after a certain episode.
                    epsilon = max(EPSILON_END,epsilon-(epsilon-EPSILON_END)/CONSTANT)
                else:
                    # If the episode number is not greater than the threshold, use the initial epsilon value.
                    epsilon = EPSILON

                # Choose an action either by exploiting the current Q-values (with probability 1 - epsilon) or
                # exploring randomly (with probability epsilon).
                if random.random() > epsilon:
                    # If exploiting, use the make_action method to get the action from the trained Q-network.
                    action = self.make_action(curr_state)
                else:
                    #  If exploring, choose a random action.
                    action = np.random.randint(0,4)


                # Take a step in the environment based on the chosen action and obtain the next state, reward, and whether the episode is done.
                next_state,reward,done,_,_=self.env.step(action)

                # Convert the reward and action to PyTorch tensors.
                tensor_reward = torch.tensor([reward], device=device)
                tensor_action = torch.tensor([action], dtype=torch.int64, device=device)

                # Preprocess the current states:
                # Convert to NumPy array, normalize pixel values, transpose dimensions, and convert to PyTorch tensors.
                state = np.asarray(curr_state, dtype=np.float32)/255
                state = state.transpose(2, 0, 1)
                store_buffer_curr_state = torch.from_numpy(state).unsqueeze(0)

                # Preprocess the next states:
                state = np.asarray(next_state,dtype=np.float32)/255
                state = state.transpose(2, 0, 1)
                store_buffer_next_state = torch.from_numpy(state).unsqueeze(0)


                # Store the transition (current state, action, reward, next state) in the replay buffer using the push method.
                self.push(store_buffer_curr_state, tensor_action, tensor_reward, store_buffer_next_state)

                # Update the current state for the next iteration, and accumulate the episode reward.
                curr_state = next_state
                epi_reward += reward

                # If the replay buffer size reaches a certain threshold, optimize the Q-network using the optimize method.
                if len(self.buffer)>=5000:
                    self.optimize()

                # If the episode is done, append the episode reward to the reward_buffer,
                # increment the episode counter, and exit the inner loop.
                if done:
                    reward_buffer.append(epi_reward)
                    episode+=1
                    break
                # Increment the timestamp counter.
                t_stamp+=1

            # Log metrics every 100 episodes using Weights & Biases (wandb)
            if episode % 100 == 0:
                wandb.log({"reward":np.mean(reward_buffer),"episode":episode,"epsilon":epsilon,"timestamp":t_stamp})

            # Update the target Q-network (self.Q_cap) at intervals specified by TARGET_UPDATE_FREQUENCY.
            if episode % TARGET_UPDATE_FREQUENCY == 0:
                self.Q_cap.load_state_dict(self.Q.state_dict())

            # Save the Q-network model at intervals specified by SAVE_MODEL_AFTER.
            if episode % SAVE_MODEL_AFTER == 0:
                # if exp_reward <= np.mean(reward_buffer):
                torch.save(self.Q.state_dict(), "final_dqn_model.pth")
                print("saving model at reward %f",np.mean(reward_buffer))
                exp_reward = np.mean(reward_buffer)
                wandb.log({"exp_reward":exp_reward})

        # Save the final Q-network model and print "Done Wooooo" when the training loop completes.
        torch.save(self.Q.state_dict(), "final_dqn_model.pth")
        print("Done Wooooo")

    def optimize(self):

        # Print a message indicating that the optimization process is starting.
        print("optimizing")
        # Retrieve a batch of transitions from the replay buffer, where each transition contains the current state, action, reward, and next state.
        transitions = self.replay_buffer(BATCH_SIZE)
        # Unpack the batch of transitions into separate lists for states, actions, rewards, and next states.
        batch = Transition(*zip(*transitions))

        # Create a boolean mask indicating which next states are not None (i.e., the states that are not terminal states).
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
        # Concatenate the non-terminal next states into a single tensor.
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        # Concatenate the current states into a single tensor.
        state_batch = torch.cat(batch.state)
        #  Concatenate the actions into a single tensor.
        action_batch = torch.cat(batch.action)
        #  Concatenate the rewards into a single tensor.
        reward_batch = torch.cat(batch.reward)

        # print(action_batch)
        # print("action batch shape")
        # print(action_batch.size())

        # Compute the Q-values for the current states using the main Q-network.
        sav = self.Q(state_batch)

        # print(sav)
        # print("sav shape")
        # print(torch.arange(sav.size(0)))
        # state_action_values=sav.gather(1, action_batch) # according to pytorch tutorial it shoul work but had to hardcode the .gather function due to some errors

        # Extract the Q-values corresponding to the taken actions from the computed Q-values.
        # Each element in state_action_values corresponds to the Q-value of the action that was taken for the respective state.
        # This is a crucial step in the Q-learning algorithm, where the Q-value for the chosen action is used in the computation of the loss during training.
        state_action_values = sav[torch.arange(sav.size(0)), action_batch]
        # print(state_action_values)

        # Initialize a tensor of zeros for the Q-values of the next states.
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        # For non-terminal next states, compute the maximum Q-value using the target Q-network (self.Q_cap) and detach it from the computation graph.
        next_state_values[non_final_mask] = self.Q_cap(non_final_next_states).max(1)[0].detach()


        # Compute the expected Q-values using the Bellman equation.
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch
        expected_state_action_values = torch.reshape(expected_state_action_values.unsqueeze(1), (1, BATCH_SIZE))[0]

        # Compute Huber loss

        # Define the Huber loss criterion.
        criterion = nn.SmoothL1Loss()
        # Compute the loss between the predicted Q-values and the expected Q-values.
        loss = criterion(state_action_values, expected_state_action_values)
        # Log the loss using Weights & Biases (wandb).
        wandb.log({"loss":loss})

        # Optimize the model

        # Zero the gradients to prevent accumulation.
        self.optimizer.zero_grad()
        # Backpropagate the loss through the network.
        loss.backward()
        for param in self.Q.parameters():
            # Clip the gradients to mitigate exploding gradients.
            param.grad.data.clamp_(-1, 1)
        # Update the model parameters using the optimizer.
        self.optimizer.step()

        # return loss

"""this method performs the optimization step for the Deep Q-Network (DQN)
using a batch of transitions from the replay buffer. It computes the loss, backpropagates
the gradients, clips the gradients, and updates the model parameters. The loss is logged using Weights & Biases."""