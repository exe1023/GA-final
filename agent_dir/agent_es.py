from es.model import Model
import es.es as es
from es.game_config import Game
from tensorboardX import SummaryWriter
import numpy as np
import scipy

def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195] # crop
    I = I[::2,::2,0] # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1    # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()


class Agent_ES:
    def __init__(self, args, env, solve):
        # environment
        self.env = env
        self.solve = solve
        # model
        game = Game(env_name=args.env_name,
                    input_size=80*80,
                    output_size=env.action_space.n,
                    time_factor=0,
                    layers=[64, 32],
                    activation='softmax',
                    noise_bias=0.0,
                    output_noise=[False, False, False])
        self.model = Model(game)
        self.model.set_env(env)
        # solver 
        self.num_params = self.model.param_count
        self.npop = args.npop
        self.solver = es.OpenES(self.num_params)
        # logger and other settings
        self.writer = SummaryWriter()
        self.num_timesteps = args.num_timesteps
        self.display_freq = args.display_freq
    
    def fitness(self, weights):
        total_reward = 0.0
        num_run = 10
        self.model.set_model_params(weights)
        for t in range(num_run):
            state = self.env.reset()
            done = False
            while(not done):
                action = self.model.get_action(prepro(state))
            
                state, reward, done, info = self.env.step(action)
                total_reward += reward
        return total_reward / num_run

    def train(self):
        for i_timestep in range(self.num_timesteps):
            solutions = self.solver.ask()
            rewards = []
            for weight in solutions:
                reward = self.fitness(weight)
            rewards = np.array(rewards)
            
            self.solver.tell(rewards)

            model_params, best, curr_best = self.solver.result()
            self.model.set_model_params(np.array(model_params).round(4))


            self.writer.add_scalar('best reward', np.max(rewards), i_timestep)
            self.writer.add_scalar('mean reward', np.mean(rewards), i_timestep)
            if i_timestep % self.display_freq == 0:
                print('Step: %d | Best Reward %f | Avg Reward %f '%
                       (i_timestep, np.max(rewards), np.mean(rewards)))
            
            if np.max(rewards) > self.solve[0]:
                print('Solve the environment. Stop training')
                print('# fitness function:%d'%(
                      (i_timestep+1) * self.npop
                ))
                break
        self.writer.close()