import argparse
import pdb
import sys
import traceback
from agent_dir.agent_dqn import AgentDQN
from agent_dir.agent_grl import AgentGRL
from utils.environment import Environment


def main(args):
    env = Environment(args.env_name)

    if len(env.observation_space.shape) > 1:
        from models.value_nets import CNNValueNet
        from models.classifiers import CNNClassifier
        value_net = CNNValueNet(env.observation_space.shape,
                                env.action_space.n)
        clf = CNNClassifier(env.observation_space.shape,
                            env.action_space.n)
    else:
        from models.value_nets import DNNValueNet
        from models.classifiers import DNNClassifier
        value_net = DNNValueNet(env.observation_space.shape,
                                env.action_space.n)
        clf = DNNClassifier(env.observation_space.shape,
                            env.action_space.n)

    agent_dqn = AgentDQN(env, value_net)
    agent_grl = AgentGRL(agent_dqn, clf,
                         population_size=args.population_size)

    agent_grl.train()


def parse_args():
    parser = argparse.ArgumentParser(description="Genetic Reinforce Learning")
    parser.add_argument('--env_name', type=str, default='Pong-v0',
                        help='Environment name.')
    parser.add_argument('--population_size', type=int, default=8)
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
