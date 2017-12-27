import argparse
from agent_dir.agent_pg import Agent_PG
from atari_wrapper import make_wrap_atari
import gym

def parse():
    parser = argparse.ArgumentParser(description="MLDS&ADL HW3")
    parser.add_argument('--test',action='store_true')
    parser.add_argument('--video_dir',default=None ,help='output video directory')
    parser.add_argument('--do_render',action='store_true' ,help='whether render environment')
    parser.add_argument('--env_name',type=str,default='PongNoFrameskip-v4' ,help='whether render environment')

    # Training hyperparameters
    parser.add_argument('--batch_size',type=int,default=32,help='batch size for training')
    parser.add_argument('--num_timesteps',type=int,default=int(10e6),help='number of training steps')
    parser.add_argument('--display_freq',type=int,default=10,help='display training status every n episodes')
    parser.add_argument('--save_freq',type=int,default=5,help='save model every n steps')
    
    

    args = parser.parse_args()
    return args

def run(args):
    #env = gym.make('CartPole-v0')
    env = gym.make('LunarLander-v2')
    #env = gym.make('Pong-v0')
    solve = (195, 100) # we solve cartpole when getting reward of 195 over 100 episode
    agent = Agent_PG(args, env, solve)
    agent.train()

if __name__ == '__main__':
    args = parse()
    run(args)