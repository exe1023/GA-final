import argparse
import pdb
import sys
import traceback
from agent_dir.agent_dqn import AgentDQN
from agent_dir.agent_grl import AgentGRL
from utils.classifier import Classifier
from utils.environment import wrap_env


def main(args):
    env = wrap_env(args.env_name)

    if len(env.observation_space.shape) > 1:
        from models.value_nets import CNNValueNet as ValueNet
        from models.clf_nets import CNNet as CLFNet
    else:
        from models.value_nets import DNNValueNet as ValueNet
        from models.clf_nets import DNNet as CLFNet

    value_net = ValueNet(env.observation_space.shape,
                         env.action_space.n)
    clf_net = CLFNet(env.observation_space.shape, 2)
    clf = Classifier(clf_net, max_epochs=5)

    agent_dqn = AgentDQN(env, value_net,
                         max_timesteps=100,
                         exploration_steps=10,
                         buffer_size=100)
    agent_grl = AgentGRL(agent_dqn, clf,
                         population_size=args.population_size,
                         n_workers=4)

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
