import argparse
from agent_dir.agent_es import Agent_ES
from atari_wrapper import make_wrap_atari
import gym

def parse():
    parser = argparse.ArgumentParser(description='ES RL')
    parser.add_argument('--env_name',type=str,default='PongNoFrameskip-v4' ,help='whether render environment')

    # Training hyperparameters
    parser.add_argument('--num_params',type=int,default=2,help='number of parameters')
    parser.add_argument('--npop',type=int,default=50,help='population size')
    parser.add_argument('--num_timesteps',type=int,default=50,help='number of training steps')
    parser.add_argument('--display_freq',type=int,default=1,help='display training status every n episodes')
    parser.add_argument('--save_freq',type=int,default=200,help='save model every n steps')
    parser.add_argument('--n_workers',type=int,default=None,help='Number of threads to do eval.')
    

    args = parser.parse_args()
    return args

def run(args):
    env = gym.make('MountainCar-v0')
    solve = (195, 100) # we solve cartpole when getting reward of 195 over 100 episode
    agent = Agent_ES(args, env, solve)
    agent.train()

if __name__ == '__main__':
    args = parse()
    run(args)
