import argparse
import pdb
import sys
import traceback
from agent_dir.agent_dqn import AgentDQN
from agent_dir.agent_grl import AgentGRL
from utils.environment import Environment


def main(args):
    env = Environment(args.env_name)

    if env.observation_space > 1:
        from models.value_nets import CNNValueNet
        value_net = CNNValueNet(env.observation_space,
                                env.action_space.n)
    else:
        from models.value_nets import DNNValueNet
        value_net = DNNValueNet(env.observation_space,
                                env.action_space.n)

    agent_dqn = AgentDQN(env, value_net)
    agent_grl = AgentGRL(env, agent_dqn, population_size=2)

    agent_grl.train()


def parse_args():
    parser = argparse.ArgumentParser(description="Genetic Reinforce Learning")
    parser.add_argument('--env_name', type=str, default='Pong-v0',
                        help='Environment name.')
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
