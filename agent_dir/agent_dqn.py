from agent_dir.agent import Agent
from dqn.module import DQN
from dqn.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from dqn.schedule import LinearSchedule
import random
import math
import numpy as np
from collections import deque

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

from tensorboardX import SummaryWriter
writer = SummaryWriter()

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
DoubleTensor = torch.cuda.DoubleTensor if use_cuda else torch.DoubleTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

def MSELoss(input, target):
   return (input.squeeze() - target.squeeze()) ** 2

class Agent_DQN(Agent):
    def __init__(self, args, env):
        self.env = env
        self.input_channels = 3 if 'SpaceInvaders' in args.env_id else 4
        self.num_actions = self.env.action_space.n

        # if testing, simply load the model we have trained
        if args.test_dqn:
            self.load(args.model)
            self.online_net.eval()
            self.target_net.eval()
            return
        # DQN variants setting
        self.prioritized = args.prioritized
        self.double = args.double
        self.n_steps = args.n_steps
        self.noise_linear = args.noise_linear
        if self.prioritized:
            self.memory = PrioritizedReplayBuffer(10000, alpha=0.6)
            self.beta_schedule = LinearSchedule(args.num_timesteps,
                                       initial_p=0.4,
                                       final_p=1.0)
            
            self.criterion = MSELoss
        else:
            self.memory = ReplayBuffer(10000)
            self.criterion = nn.MSELoss()
        
        # build target, online network
        self.target_net = DQN(self.input_channels, 
                              self.num_actions,
                              dueling=args.dueling,
                              noise_linear=args.noise_linear)
        self.target_net = self.target_net.cuda() if use_cuda else self.target_net
        self.online_net = DQN(self.input_channels, 
                              self.num_actions,
                              dueling=args.dueling,
                              noise_linear=args.noise_linear)
        self.online_net = self.online_net.cuda() if use_cuda else self.online_net
        
        # discounted reward
        self.GAMMA = 0.99
        
        # exploration setting 
        self.exploration = LinearSchedule(schedule_timesteps=int(0.1 * args.num_timesteps),
                                 initial_p=1.0,
                                 final_p=0.05)

        # training settings
        self.train_freq = 4
        self.learning_start = 10000
        self.batch_size = args.batch_size
        self.num_timesteps = args.num_timesteps
        self.display_freq = args.display_freq
        self.save_freq = args.save_freq
        self.target_update_freq = args.target_update_freq
        self.optimizer = optim.RMSprop(self.online_net.parameters(),
                                      lr=1e-4)
        # global status
        self.episodes_done = 0
        self.steps = 0
    
    def make_action(self, observation, test=True):
        return self.act(observation, test)
    
    def save(self, save_path):
        print('save model to', save_path)
        torch.save(self.online_net, save_path + '_online')
        torch.save(self.target_net, save_path + '_target')
    
    def load(self, load_path):
        if use_cuda:
            self.online_net = torch.load(load_path + '_online')
            self.target_net = torch.load(load_path + '_target')
        else:
            self.online_net = torch.load(load_path + '_online', map_location=lambda storage, loc: storage)
            self.target_net = torch.load(load_path + '_target', map_location=lambda storage, loc: storage)

    def act(self, state, test=False):
        sample = random.random()
        if test:
            eps_threshold = 0.01
            state = torch.from_numpy(state).permute(2, 0, 1).unsqueeze(0)
            state = state.cuda() if use_cuda else state
        else:
            eps_threshold = self.exploration.value(self.steps)
        
        if sample > eps_threshold:
            action = self.online_net(
                Variable(state, volatile=True).type(FloatTensor)).data.max(1)[1].view(1, 1)
        else:
            action = LongTensor([[random.randrange(self.num_actions)]])
        return action if not test else action[0, 0]
    
    def reset_noise(self):
        assert self.noise_linear == True
        self.online_net.reset_noise()
        self.target_net.reset_noise()

    def update(self):
        if self.prioritized:
            batch, weight, batch_idxes = self.memory.sample(self.batch_size, 
                                                     beta=self.beta_schedule.value(self.steps))
            weight_batch = Variable(Tensor(weight)).squeeze()
        else:
            batch = self.memory.sample(self.batch_size)
        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = ByteTensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)))

        # We don't want to backprop through the expected action values and volatile
        # will save us on temporarily changing the model parameters'
        # requires_grad to False!
        non_final_next_states = Variable(torch.cat([s for s in batch.next_state
                                                    if s is not None]),
                                        volatile=True)
        state_batch = Variable(torch.cat(batch.state))
        action_batch = Variable(torch.cat(batch.action))
        reward_batch = Variable(torch.cat(batch.reward))

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = self.online_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = Variable(torch.zeros(self.batch_size).type(Tensor))
        q_next = self.target_net(non_final_next_states)
        if self.double:
            _, best_actions = self.online_net(non_final_next_states).max(1)
            next_state_values[non_final_mask] = q_next.gather(1, best_actions.unsqueeze(1)).squeeze(1)
        else:
            next_state_values[non_final_mask] = q_next.max(1)[0]

        # Now, we don't want to mess up the loss with a volatile flag, so let's
        # clear it. After this, we'll just end up with a Variable that has
        # requires_grad=False
        next_state_values.volatile = False
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * (self.GAMMA ** (self.n_steps))) + reward_batch

        # Compute loss
        if self.prioritized:
            loss = self.criterion(state_action_values, expected_state_action_values)
            loss = torch.mul(loss,weight_batch)
            new_priorities = np.abs(loss.cpu().data.numpy()) + 1e-6
            self.memory.update_priorities(batch_idxes, new_priorities)
            loss = loss.mean()
        else:
            loss = self.criterion(state_action_values, expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.data[0]

    def train(self):
        total_reward = 0
        loss = 0
        # set training mode
        self.online_net.train()
        while(True):
            if self.noise_linear:
                self.reset_noise()
            state = np.array(self.env.reset())
            # map shape: (84,84,4) --> (1,4,84,84)
            state = torch.from_numpy(state).permute(2,0,1).unsqueeze(0)
            state = state.cuda() if use_cuda else state
            done = False
            episode_duration = 0
            while(not done):
                # select and perform action
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action[0, 0])
                total_reward += reward
                reward = Tensor([reward])

                # process new state 
                next_state = torch.from_numpy(np.array(next_state)).permute(2,0,1).unsqueeze(0)
                next_state = next_state.cuda() if use_cuda else next_state
                if done:
                    next_state = None
                
                # store the transition in memory
                self.memory.push(state, action, next_state, reward)

                # move to the next state
                state = next_state

                # Perform one step of the optimization (on the target network)
                if self.steps > self.learning_start and self.steps % self.train_freq == 0:
                    loss = self.update()
                    if self.noise_linear:
                        self.reset_noise()
                
                # update target network
                if self.steps > self.learning_start and self.steps % self.target_update_freq == 0:
                    self.target_net.load_state_dict(self.online_net.state_dict())
                
                if self.steps % self.save_freq == 0:
                    self.save('dqn.cpt')
            
                self.steps += 1
                episode_duration += 1
            
            if self.episodes_done % self.display_freq == 0:
                print('Episode: %d | Steps: %d/%d | Exploration: %f | Avg reward: %f | loss: %f | Episode Duration: %d'%
                        (self.episodes_done, self.steps, self.num_timesteps, self.exploration.value(self.steps), total_reward / self.display_freq, 
                        loss, episode_duration))
                writer.add_scalar('reward', total_reward/self.display_freq, self.steps)
                total_reward = 0
            
            self.episodes_done += 1
            if self.steps > self.num_timesteps:
                break
        self.save('dqn_final.model')

    def nsteps_train(self):
        '''
        Training procedure for multi-steps learning
        '''
        total_reward = 0
        loss = 0
        while(True):
            state_buffer = deque() # store states for future use
            action_buffer = deque() # store actions for future use
            reward_buffer = deque() # store rewards for future use
            nstep_reward = 0 # calculate n-step discounted reward
            state = self.env.reset()
            # map shape: (84,84,4) --> (1,4,84,84)
            state = torch.from_numpy(state).permute(2,0,1).unsqueeze(0)
            state = state.cuda() if use_cuda else state
            state_buffer.append(state)

            done = False
            episode_duration = 0

            # run n-1 steps first
            for _ in range(1, self.n_steps):
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action[0, 0])
                next_state = torch.from_numpy(next_state).permute(2,0,1).unsqueeze(0)
                next_state = next_state.cuda() if use_cuda else next_state
                if done:
                    next_state = None
                state_buffer.append(next_state)
                action_buffer.append(action)
                nstep_reward = nstep_reward * self.GAMMA + reward
                reward_buffer.append(reward)

                state = next_state
                episode_duration += 1

            while(not done):
                # select and perform action
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action[0, 0])
                total_reward += reward

                # process new state 
                next_state = torch.from_numpy(next_state).permute(2,0,1).unsqueeze(0)
                next_state = next_state.cuda() if use_cuda else next_state
                if done:
                    next_state = None
                
                # save new state, action, reward
                state_buffer.append(next_state)
                action_buffer.append(action)
                reward_buffer.append(reward)
                nstep_reward = nstep_reward * self.GAMMA + reward
 
                # store the transition in memory
                self.memory.push(state_buffer.popleft(), action_buffer.popleft(), next_state, Tensor([nstep_reward]) )

                # update n-step reward
                nstep_reward -= (self.GAMMA ** (self.n_steps-1)) * reward_buffer.popleft()

                # move to the next state
                state = next_state

                # Perform one step of the optimization (on the target network)
                if self.steps > self.learning_start and self.steps % self.train_freq == 0:
                    loss = self.update()
                
                # update target network
                if self.steps > self.learning_start and self.steps % self.target_update_freq == 0:
                    self.target_net.load_state_dict(self.online_net.state_dict())
                
                if self.steps % self.save_freq == 0:
                    self.save('dqn.cpt')
            
                self.steps += 1
                episode_duration += 1
            
            if self.episodes_done % self.display_freq == 0:
                print('Episode: %d | Steps: %d/%d | Exploration: %f | Avg reward: %f | loss: %f | Episode Duration: %d'%
                        (self.episodes_done, self.steps, self.num_timesteps, self.exploration.value(self.steps), total_reward / self.display_freq, 
                        loss, episode_duration))
                total_reward = 0
            
            self.episodes_done += 1
            if self.steps > self.num_timesteps:
                break
        self.save('dqn_final.model')
