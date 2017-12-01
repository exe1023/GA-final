import copy
import multiprocessing
from es.simple_es import ES
from tensorboardX import SummaryWriter
import numpy as np


class Agent_ES:
    def __init__(self, args, env, solve):
        # environment
        self.env = env
        self.solve = solve
        # solver
        self.num_params = args.num_params
        self.npop = args.npop
        self.solver = ES(args.num_params,
                         args.npop)
        # logger and other settings
        self.writer = SummaryWriter()
        self.num_timesteps = args.num_timesteps
        self.display_freq = args.display_freq

        # parallelism args
        self.n_workers = args.n_workers or multiprocessing.cpu_count()

    @staticmethod
    def fitness(args):
        env, weight = args
        total_reward = 0.0
        num_run = 100
        for t in range(num_run):
            observation = env.reset()
            done = False
            while(not done):
                action = 2 if np.dot(weight, observation) > 0 else 0

                observation, reward, done, info = env.step(action)
                total_reward += reward
        return total_reward / num_run

    def train(self):
        for i_timestep in range(self.num_timesteps):
            solutions = self.solver.ask()

            with multiprocessing.Pool(self.n_workers) as p:
                envs = [copy.deepcopy(self.env) for i in range(len(solutions))]
                rewards = p.map(self.fitness, zip(envs, solutions))
                rewards = list(rewards)
            rewards = np.array(rewards)
            self.solver.tell(rewards)

            self.writer.add_scalar('best reward', np.max(rewards), i_timestep)
            self.writer.add_scalar('mean reward', np.mean(rewards), i_timestep)
            if i_timestep % self.display_freq == 0:
                print('Step: %d | Best Reward %f | Avg Reward %f | Weight %s' %
                      (i_timestep, np.max(rewards), np.mean(rewards), self.solver.solution))

            if np.max(rewards) > self.solve[0]:
                print('Solve the environment. Stop training')
                print('# fitness function:%d' % (
                      (i_timestep + 1) * self.npop
                      ))
                break
        self.writer.close()
