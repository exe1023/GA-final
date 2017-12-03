import copy
import numpy as np
import random
import torch
from torch.autograd import Variable
from agent_base import AgentBase
from openai_replay_buffer import PrioritizedReplayBuffer as ReplayBuffer


class AgentDQN(AgentBase):
    """A DQN agent

    Args:
        env: Env made with OpenAI gym.
        model (Torch.nn.Module): A nn module that map observation from
            env and action to values.
        max_timesteps (int): Number of update to run,
        gamma (int): Gamma in the Bellman equation.
        batch_size (int): Number of examples to use when doing gradient
            descend.
        learning_rate (float): Learning rate when doing gradient descend.
        exploration_steps (int): Number of exploration steps.
        exploration_final_eps (float): Exploration rate after
            exploration_steps.
        prioritized_replay_eps (float): Minimum priority a sample will have.
        prioritized_replay_alpha (float): How strong the sample probability
            related to priority.
        buffer_size (int): Size of replay buffer.
        target_network_update_period (int): Period of update target network
            with online network.
    """
    def __init__(self, env, model,
                 max_timesteps=50000,
                 gamma=0.999,
                 batch_size=32,
                 learning_rate=1e-3,
                 exploration_steps=5000,
                 exploration_final_eps=0.1,
                 buffer_size=50000,
                 prioritized_replay_eps=1e-4,
                 prioritized_replay_alpha=0.9,
                 target_network_update_period=5000,
                 log_file=None):
        self.t = 0
        self.env = env
        self.n_actions = self.env.action_space.n
        self.batch_size = batch_size
        self.max_timesteps = max_timesteps
        self.gamma = gamma
        self.exploration_steps = exploration_steps
        self.exploration_final_eps = exploration_final_eps
        self.replay_buffer = ReplayBuffer(int(buffer_size),
                                          prioritized_replay_alpha)
        self.prioritized_replay_eps = prioritized_replay_eps
        self.target_network_update_period = target_network_update_period
        self.log_file = log_file

        self.model = model
        self.use_cuda = torch.cuda.is_available()
        self.optimizer = torch.optim.Adam(self._model.parameters(),
                                          lr=learning_rate)
        if self._use_cuda:
            self.model = self._model.cuda()

    def make_action(self, state, test=True):

        # decide if doing exploration
        if not test:
            self.epsilon = 1 \
                - (1 - self.exploration_final_eps) \
                * self.t / self.exploration_steps
            self.epsilon = max(self.epsilon, self.exploration_final_eps)
        else:
            self.epsilon = self.exploration_final_eps
        explore = random.random() < self.epsilon

        if explore:
            return random.randint(0, self.n_actions - 1)
        else:
            state = torch.from_numpy(state).float()
            state = Variable(state, volatile=True)
            if self._use_cuda:
                state = state.cuda()
            action_value = self._model.forward(state.unsqueeze(0))
            best_action = action_value.max(-1)[1].data.cpu().numpy()
            return best_action[0]

    def update_model(self, target_q):
        # sample from replay_buffer
        beta = 0.1 \
            + (1 - 0.10) \
            * self.t / self.max_timesteps
        beta = min(beta, 1)
        replay = self.replay_buffer.sample(self.batch_size, beta=beta)

        # prepare tensors
        tensor_replay = [torch.from_numpy(val) for val in replay]
        if self._use_cuda:
            tensor_replay = [val.cuda() for val in tensor_replay]
        states0, actions, rewards, states1, dones, \
            _, weights = tensor_replay

        # predict target with target network
        var_states1 = Variable(states1.float())
        var_target_reward = \
            target_q.forward(var_states1).max(-1)[0]
        var_targets = Variable(rewards) \
            + self.gamma * var_target_reward * (-Variable(dones) + 1)
        var_targets = var_targets.unsqueeze(-1).detach()

        # gradient descend model
        var_states0 = Variable(states0.float())
        var_action_values = self._model.forward(var_states0) \
            .gather(1, Variable(actions.view(-1, 1)))
        var_loss = (var_action_values - var_targets) ** 2

        if self.t % 5000 == 0:
            print(var_targets)

        # weighted sum loss
        var_weights = Variable(weights)
        var_loss_mean = torch.sum(var_loss * var_weights) / self.batch_size

        # gradient descend loss
        self._optimizer.zero_grad()
        var_loss_mean.backward()
        self._optimizer.step()

        # update experience priorities
        indices = replay[5]
        loss = torch.abs(var_action_values - var_targets).data.cpu().numpy()
        new_priority = loss + self.prioritized_replay_eps
        self.replay_buffer.update_priorities(indices, new_priority)

        return np.mean(loss)

    def train(self):
        # init target network
        target_q = copy.deepcopy(self.model)

        # Get first state
        state0 = self.env.reset()

        # log statics
        loss = 0
        episode_rewards = [0]
        best_mean_reward = 0

        # make log file pointer
        if self.log_file is not None:
            fp_log = open(self.log_file, 'w', buffering=1)

        while self.t < self.max_timesteps:
            # play
            for i in range(4):
                action = self.make_action(state0, False)
                state1, reward, done, _ = self.env.step(action)
                self.replay_buffer.add(state0, action,
                                       float(reward), state1, float(done))
                # accumulate episode reward
                episode_rewards[-1] += reward

                # update previous state and log
                if done:
                    state0 = self.env.reset()
                    print('t = %d, r = %f, loss = %f, exp = %f'
                          % (self.t, episode_rewards[-1], loss, self.epsilon))
                    if self.log_file is not None:
                        fp_log.write('{},{},{}\n'.format(self.t,
                                                         episode_rewards[-1],
                                                         loss))
                    episode_rewards.append(0)
                else:
                    state0 = state1

            loss = self.update_model(target_q)

            # update target network
            if self.t % self.target_network_update_freq == 0:
                target_q.load_state_dict(self._model.state_dict())

            if self.t % self.save_freq == 0:
                mean_reward = \
                    sum(episode_rewards[-100:]) / len(episode_rewards[-100:])
                if best_mean_reward < mean_reward:
                    print('save best model with mean reward = %f'
                          % mean_reward)
                    best_mean_reward = mean_reward
                    torch.save({
                        'model': self._model.state_dict()
                    }, 'model')

            self.t += 1
