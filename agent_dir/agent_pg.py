import numpy as np
from pg.module import REINFORCE
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

use_cuda = torch.cuda.is_available()

class Agent_PG:
    def __init__(self,args, env, solve):
        # environment
        self.env = env
        self.solve = solve
        # model 
        self.num_actions = self.env.action_space.n
        self.model = REINFORCE(env.observation_space.shape[0], self.num_actions)
        self.model = self.model.cuda() if use_cuda else self.model
        # training settings
        self.batch_size = args.batch_size # not used in reinforce
        self.num_timesteps = args.num_timesteps
        self.display_freq = args.display_freq
        self.save_freq = args.save_freq
        self.optimizer = optim.Adam(self.model.parameters(),
                                       lr=1e-4)
        self.GAMMA = 0.99
        # logger
        self.writer = SummaryWriter()
        
        

    def select_action(self, state):
        state = Variable(state).cuda() if use_cuda else Variable(state)
        probs = self.model(state) 
        action = probs.multinomial().data
        prob = probs[:, action[0,0]].view(1, -1)
        log_prob = prob.log()
        entropy = - (probs*probs.log()).sum()

        return action[0], log_prob, entropy
    
    def save(self, save_path):
        print('save model to', save_path)
        torch.save(self.model, save_path)
    
    def load(self, load_path):
        if use_cuda:
            self.model = torch.load(load_path)
        else:
            self.model = torch.load(load_path, map_location=lambda storage, loc: storage)
    
    def update(self, rewards, log_probs, entropies, gamma):
        # compute loss with entropy regularization
        R = torch.zeros(1, 1)
        loss = 0
        for i in reversed(range(len(rewards))):
            R = gamma * R + rewards[i]
            loss = loss - (log_probs[i]*(Variable(R).expand_as(log_probs[i])).cuda()).sum() - (0.0001*entropies[i].cuda()).sum()
        loss = loss / len(rewards)
		
        # run backprop
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(self.model.parameters(), 40)
        self.optimizer.step()
        return loss.data[0]

    def train(self):
        # accumulate reward for display
        total_reward = 0
        # accumulate reward for testing solve criterion
        solve_reward = 0
        for i_episode in range(self.num_timesteps):
            # map shape: (84,84,4) -> (1,4,84,84)
            #state = torch.from_numpy(self.env.reset()).permute(2,0,1).unsqueeze(0)
            state = torch.Tensor([self.env.reset()])
            entropies = []
            log_probs = []
            rewards = []
            done = False
            while(not done):
                action, log_prob, entropy = self.select_action(state)
                action = action.cpu() if use_cuda else action

                next_state, reward, done, _ = self.env.step(action.numpy()[0])

                entropies.append(entropy)
                log_probs.append(log_prob)
                rewards.append(reward)
                #next_state = torch.from_numpy(next_state).permute(2,0,1).unsqueeze(0)
                next_state = torch.Tensor([next_state])
                state = next_state
                
                total_reward += reward
                solve_reward += reward
            loss = self.update(rewards, log_probs, entropies, self.GAMMA)
            
            # log
            self.writer.add_scalar('reward', np.sum(rewards), i_episode)
            self.writer.add_scalar('loss', loss, i_episode)
            
            if i_episode % self.display_freq == 0:
                print('Episode: %d/%d | Avg reward: %f | loss: %f'%
                    (i_episode, self.num_timesteps, total_reward / self.display_freq, loss))
                total_reward = 0
            
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