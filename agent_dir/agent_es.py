from es.model import Model
import es.es as es
from es.game_config import Game
from tensorboardX import SummaryWriter
import scipy
from tqdm import tqdm
import multiprocessing
import numpy as np
#import cupy as np

def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195] # crop
    I = I[::2,::2,0] # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1    # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()

class Seeder:
    def __init__(self, init_seed=0):
        np.random.seed(init_seed)
        self.limit = np.int32(2**31-1)
    def next_seed(self):
        result = np.random.randint(self.limit)
        return result
    def next_batch(self, batch_size):
        result = np.random.randint(self.limit, size=batch_size).tolist()
        return result

def fitness(env, model, weights, seed):
    total_reward = 0.0
    #num_run = 1
    num_run = 5
    model.set_model_params(weights)
        
    env.seed(seed)

    for t in range(num_run):
        state = env.reset()
        done = False
        while(not done):
            action = model.get_action(state)
            #action = model.get_action(prepro(state))
            state, reward, done, info = env.step(action)
            total_reward += reward
    return total_reward / num_run


class Agent_ES:
    def __init__(self, args, env, solve):
        # environment
        self.env = env
        self.solve = solve
        self.args = args
        # model
        '''game = Game(env_name=args.env_name,
                    input_size=80*80,
                    output_size=env.action_space.n,
                    time_factor=0,
                    layers=[32, 16],
                    activation='softmax',
                    noise_bias=1,
                    output_noise=[False, False, False])
        '''
        
        game = Game(env_name=args.env_name,
                    input_size=env.observation_space.shape[0],
                    output_size=env.action_space.n,
                    time_factor=0,
                    layers=[64, 32],
                    activation='softmax',
                    noise_bias=1,
                    output_noise=[False, False, False])
        
        self.model = Model(game)
        self.model.set_env(env)
        # solver 
        self.num_params = self.model.param_count
        print('params:', self.num_params)
        self.npop = args.npop
        if args.solver == 'cmaes':
            self.solver = es.CMAES(self.num_params,
                                    popsize=args.npop,
                                    sigma_init=0.2)
        elif args.solver == 'openes':
            self.solver = es.OpenES(self.num_params,
                                    popsize=args.npop)

        
        # other settings
        self.writer = SummaryWriter()
        self.seeder = Seeder()
        self.num_timesteps = args.num_timesteps
        self.display_freq = args.display_freq
        self.n_workers = multiprocessing.cpu_count()
    
    def fitness(self, weights, seed):
        total_reward = 0.0
        num_run = 5
        self.model.set_model_params(weights)
        
        self.env.seed(seed)

        for t in range(num_run):
            state = self.env.reset()
            done = False
            while(not done):
                action = self.model.get_action(state)
                state, reward, done, info = self.env.step(action)
                total_reward += reward
        return total_reward / num_run

    def load(self, path='es.cpt'):
        model_params = np.load(path)
        self.model.set_model_params(model_params)
    
    def test_play(self):
        while True:
            state = self.env.reset()
            self.env.render()
            done = False
            rewards = 0
            while(not done):
                action = self.model.get_action(state)
                state, reward, done, info = self.env.step(action)
                self.env.render()
                rewards += reward
            print(rewards)

    def train(self):
        print('Start Training')
        for i_timestep in range(self.num_timesteps):
            solutions = self.solver.ask()
            seeds = self.seeder.next_batch(self.npop)
            multiple_res = []
            rewards = []
            
            if self.args.parallel:
                with multiprocessing.Pool(processes=self.n_workers) as p:
                    for weight, seed in zip(solutions, seeds):
                        res = p.apply_async(fitness, (self.env, self.model, weight, seed))
                        multiple_res.append(res)
                    p.close()
                    p.join()
                rewards = [res.get() for res in multiple_res]
            else:
                for weight, seed in zip(solutions, seeds):
                    rewards.append(fitness(self.env, self.model, weight, seed))
            rewards = np.array(rewards)
            
            self.solver.tell(rewards)

            model_params, best, curr_best, _ = self.solver.result()
            self.model.set_model_params(np.array(model_params).round(4))

            self.writer.add_scalar('best reward', np.max(rewards), i_timestep)
            self.writer.add_scalar('mean reward', np.mean(rewards), i_timestep)
            if i_timestep % self.display_freq == 0:
                print('Generation: %d | Best Reward %f | Avg Reward %f '%
                       (i_timestep, np.max(rewards), np.mean(rewards)))
            
            if i_timestep % self.args.save_freq == 0:
                print('save model')
                np.save('es.cpt', np.array(model_params).round(4))
            
            if np.max(rewards) > self.solve[0]:
                print('Solve the environment. Stop training')
                print('# fitness function:%d'%(
                      (i_timestep+1) * self.npop
                ))
                break
        self.writer.close()
