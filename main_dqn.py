import argparse
from test import *
import gym
from atari_wrapper import make_wrap_atari

def parse():
    parser = argparse.ArgumentParser(description="MLDS&ADL HW3")
    parser.add_argument('--train',action='store_true' ,help='whether train')
    parser.add_argument('--test_dqn',action='store_true' ,help='whether test policy gradient')
    parser.add_argument('--video_dir',default=None ,help='output video directory')
    parser.add_argument('--render',action='store_true' ,help='whether render environment')
    parser.add_argument('--env_id',type=str,default='BreakoutNoFrameskip-v4' ,help='enviornment name')
    
    # DQN model architecture
    parser.add_argument('--prioritized',action='store_true',help='whether to use prioritized replay')
    parser.add_argument('--dueling',action='store_true',help='whether to use dueling network')
    parser.add_argument('--double',action='store_true',help='whether to use double q network')
    parser.add_argument('--n_steps',type=int,default=1,help='n-steps update')

    # Training hyperparameters
    parser.add_argument('--batch_size',type=int,default=32,help='batch size for training')
    parser.add_argument('--num_timesteps',type=int,default=int(10e6),help='number of training steps')
    parser.add_argument('--display_freq',type=int,default=10,help='display training status every n episodes')
    parser.add_argument('--save_freq',type=int,default=200000,help='save model every n steps')
    parser.add_argument('--target_update_freq',type=int,default=1000,help='update target network every n episodes')
    
    args = parser.parse_args()
    return args

def run(args):
    if args.train:
        from agent_dir.agent_dqn import Agent_DQN
        env = make_wrap_atari(args.env_id, clip_rewards=True)
        agent = Agent_DQN(args, env)
        if args.n_steps > 1:
            agent.nsteps_train()
        else:
            agent.train()

if __name__ == '__main__':
    args = parse()
    run(args)