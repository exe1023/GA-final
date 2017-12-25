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
                         max_timesteps=args.gen_timesteps,
                         exploration_steps=args.exploration_steps,
                         buffer_size=5000)
    agent_grl = AgentGRL(agent_dqn, clf,
                         population_size=args.population_size,
                         n_workers=args.n_workers,
                         log_dir=args.log_dir,
                         ckp_dir=args.ckp_dir)

    agent_grl.train()


def parse_args():
    parser = argparse.ArgumentParser(description="Genetic Reinforce Learning")
    parser.add_argument('--env_name', type=str, default='Pong-v0',
                        help='Environment name.')
    parser.add_argument('--population_size', type=int, default=8)
    parser.add_argument('--gen_timesteps', type=int, default=10000)
    parser.add_argument('--n_workers', type=int, default=1)
    parser.add_argument('--log_dir', type=str, default=None,
                        help='Directory to save training log of generations.')
    parser.add_argument('--ckp_dir', type=str, default=None,
                        help='Directory to save checkpoint of generations.')
    parser.add_argument('--exploration_steps', type=int, default=25000,
                        help='Number of exploration steps for DQN to do.')
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
