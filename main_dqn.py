import argparse
import pdb
import sys
import traceback
from agent_dir.agent_dqn import AgentDQN
from utils.environment import wrap_env


class CallbackLog:
    def __init__(self, filename):
        self._fp = open(filename, 'w', buffering=1)

    def on_episode_end(self, model, rewards, loss):
        self._fp.write('{},{},{}\n'
                       .format(model.t, rewards[-1], loss))


def main(args):
    env = wrap_env(args.env_name)

    if len(env.observation_space.shape) > 1:
        from models.value_nets import CNNValueNet as ValueNet
    else:
        from models.value_nets import DNNValueNet as ValueNet

    value_net = ValueNet(env.observation_space.shape,
                         env.action_space.n)
    agent_dqn = AgentDQN(env, value_net,
                         max_timesteps=1000000,
                         exploration_steps=10000,
                         buffer_size=10000)
    callback_log = CallbackLog(args.log_file)
    agent_dqn.train([callback_log.on_episode_end])


def parse_args():
    parser = argparse.ArgumentParser(description="Genetic Reinforce Learning")
    parser.add_argument('--env_name', type=str, default='Pong-v0',
                        help='Environment name.')
    parser.add_argument('--population_size', type=int, default=8)
    parser.add_argument('--log_file', type=str, default='log-dqn.txt')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    try:
        main(args)
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
