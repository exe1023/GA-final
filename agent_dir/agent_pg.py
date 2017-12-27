import numpy as np
from pg.module import REINFORCE, REINFORCE_ATARI
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
import scipy.misc

use_cuda = torch.cuda.is_available()


def prepro(o,image_size=[80,80]) :
    # obsv : [210, 180, 3] HWC
    # preprocessing code is from https://github.com/hiwonjoon/tf-a3c-gpu/blob/master/async_agent.py
    y = 0.2126 * o[:, :, 0] + 0.7152 * o[:, :, 1] + 0.0722 * o[:, :, 2]
    y = y.astype(np.uint8)
    resized = scipy.misc.imresize(y, image_size)
    return np.expand_dims(resized.astype(np.float32),axis=0)


class Agent_PG:
    def __init__(self,args, env, solve):
        # environment
        self.env = env
        self.solve = solve
        # model 
        self.num_actions = self.env.action_space.n
        #self.model = REINFORCE_ATARI(1, self.num_actions)
        self.model = REINFORCE(env.observation_space.shape[0], self.num_actions)
        self.model = self.model.cuda() if use_cuda else self.model
        # training settings
        self.batch_size = args.batch_size # not used in reinforce
        self.num_timesteps = args.num_timesteps
        self.display_freq = args.display_freq
        self.save_freq = args.save_freq
        self.optimizer = optim.RMSprop(self.model.parameters(),
                                       lr=1e-4,
                                       alpha=0.99,
                                       eps=1e-10)
        
        '''self.optimizer = optim.Adam(self.model.parameters(),
                                      lr=1e-5,
                                      weight_decay=0.99)
        '''
        self.GAMMA = 0.99
        # logger
        self.writer = SummaryWriter()
        
        

    def select_action(self, state):
        state = Variable(state).cuda() if use_cuda else Variable(state)
        probs = self.model(state)
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        log_prob = m.log_prob(action)
        entropy = - (probs*probs.log()).sum()

        return action, log_prob, entropy
    
    def save(self, save_path):
        print('save model to', save_path)
        torch.save(self.model, save_path)
    
    def load(self, load_path):
        if use_cuda:
            self.model = torch.load(load_path)
        else:
            self.model = torch.load(load_path, map_location=lambda storage, loc: storage)
    
    def update(self, rewards, log_probs, entropies, gamma):
        R = 0.0
        for i in reversed(range(len(rewards))):
            if rewards[i] != 0:
                R = 0
            R = rewards[i] + R * 0.99
            rewards[i] = R
        rewards = np.array(rewards)
        rewards = (rewards - np.mean(rewards))/np.std(rewards)
        
        loss = 0
        for i in reversed(range(len(rewards))):
            loss -= (log_probs[i]* rewards[i])# - (0.0001*entropies[i].cuda()).sum()
        #loss = loss / len(rewards)
		
        # run backprop
        self.optimizer.zero_grad()
        loss.backward()
        #nn.utils.clip_grad_norm(self.model.parameters(), 1)
        self.optimizer.step()
        return loss.data[0]

    def train(self):
        # accumulate reward, loss for display
        total_reward = 0
        total_loss = 0
        # accumulate reward for testing solve criterion
        solve_reward = 0
        for i_episode in range(self.num_timesteps):
            state = self.env.reset()
            #state = torch.from_numpy(prepro(state)).unsqueeze(0)
            state = torch.Tensor(state).unsqueeze(0)
            prev_state = state
            entropies = []
            log_probs = []
            rewards = []
            done = False
            while(not done):
                action, log_prob, entropy = self.select_action(state)
                action = action.cpu() if use_cuda else action
                next_state, reward, done, _ = self.env.step(action.data[0])

                entropies.append(entropy)
                log_probs.append(log_prob)
                rewards.append(reward)
                #next_state = torch.from_numpy(prepro(next_state)).unsqueeze(0)
                next_state = torch.Tensor(next_state).unsqueeze(0)
                #state = next_state - prev_state
                prev_state = next_state
                
                total_reward += reward
                solve_reward += reward
            loss = self.update(rewards, log_probs, entropies, self.GAMMA)
            total_loss += loss
            # log
            self.writer.add_scalar('reward', np.sum(rewards), i_episode)
            self.writer.add_scalar('loss', loss, i_episode)
            
            if i_episode % self.display_freq == 0:
                print('Episode: %d/%d | Avg reward: %f | loss: %f'%
                    (i_episode, self.num_timesteps, total_reward / self.display_freq, total_loss / self.display_freq))
                total_reward = 0
                total_loss = 0
            
            if i_episode % self.solve[1] == 0 and i_episode > 0:
                if solve_reward / i_episode > self.solve[0]:
                    print('Solve the environment. Stop training')
                    break
            
            if i_episode % self.save_freq == 0:
                self.save('pg.cpt')
        self.save('pg.final')
        # export scalar data to JSON for external processing
        self.writer.export_scalars_to_json("./all_scalars.json")
        self.writer.close()