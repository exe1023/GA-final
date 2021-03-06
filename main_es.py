import argparse
from agent_dir.agent_es import Agent_ES
from atari_wrapper import make_wrap_atari
import gym
from custom_envs import trap

def parse():
    parser = argparse.ArgumentParser(description='ES RL')
    parser.add_argument('--env_name',type=str,default='PongNoFrameskip-v4' ,help='whether render environment')

    # Training hyperparameters
    parser.add_argument('--solver', type=str, default='cmaes')
    parser.add_argument('--num_params',type=int,default=4,help='number of parameters')
    parser.add_argument('--npop',type=int,default=50,help='population size')
    parser.add_argument('--num_timesteps',type=int,default=50,help='number of training steps')
    parser.add_argument('--display_freq',type=int,default=1,help='display training status every n episodes')
    parser.add_argument('--save_freq',type=int,default=5,help='save model every n steps')
    parser.add_argument('--parallel', type=int, default=1)
    
    parser.add_argument('--test', action='store_true')

    args = parser.parse_args()
    return args

def run(args):
    #env = gym.make('Acrobot-v1')
    #env = gym.make('MountainCar-v0')
    #env = gym.make('BipedalWalker-v2')
    env = gym.make('LunarLander-v2')
    #env = trap.MKTrap(m=20, k=5)
    solve = (200, 100) # we solve cartpole when getting reward of 195 over 100 episode
    agent = Agent_ES(args, env, solve)
    if args.test:
        agent.load('es.final.npy')
        agent.test_play()
    else:
        agent.train()

if __name__ == '__main__':
    args = parse()
    run(args)
