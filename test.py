"""

### NOTICE ###
You DO NOT need to upload this file

"""

import argparse
import numpy as np
import gym
from atari_wrapper import make_wrap_atari

seed = 11037

def parse():
    parser = argparse.ArgumentParser(description="MLDS&ADL HW3")
    parser.add_argument('--test_pg', action='store_true', help='whether test policy gradient')
    parser.add_argument('--test_dqn', action='store_true', help='whether test DQN')
    parser.add_argument('--model', type=str, default='dqn_final.model', help='model name')
    parser.add_argument('--env_id',type=str,default='BreakoutNoFrameskip-v4' ,help='enviornment name')
    parser.add_argument('--video_dir', default=None, help='output video directory')
    parser.add_argument('--do_render', action='store_true', help='whether render environment')
    try:
        from argument import add_arguments
        parser = add_arguments(parser)
    except:
        pass
    args = parser.parse_args()
    return args


def test(agent, env, total_episodes=30):
    rewards = []
    env.seed(seed)
    for i in range(total_episodes):
        state = np.array(env.reset())
        agent.init_game_setting()
        done = False
        episode_reward = 0.0

        #playing one game
        while(not done):
            action = agent.make_action(state, test=True)
            state, reward, done, info = env.step(action)
            episode_reward += reward
            state = np.array(state)

        rewards.append(episode_reward)
    print('Run %d episodes'%(total_episodes))
    print('Mean:', np.mean(rewards))


def run(args):
    if args.test_pg:
        env = gym.make('Pong-v0')
        from agent_dir.agent_pg import Agent_PG
        agent = Agent_PG(env, args)
        test(agent, env)

    if args.test_dqn:
        env = make_wrap_atari(args.env_id, clip_rewards=False)
        from agent_dir.agent_dqn import Agent_DQN
        agent = Agent_DQN(args, env)
        test(agent, env, total_episodes=100)


if __name__ == '__main__':
    args = parse()
    run(args)
