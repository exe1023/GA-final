import copy
import numpy as np
import random
import torch
from torch.autograd import Variable
from .agent_base import AgentBase
from utils.openai_replay_buffer import PrioritizedReplayBuffer as ReplayBuffer


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
        distil_dagger_iters (int): Number of iterations to run DAgger.
        distil_epochs (int): Number epochs to run when fitting self.model with
            expert's prediction.
        dagger_explore_steps (int): Number of steps to explore in each dagger
            iterations.
        save_path (str): Path to store model.
        save_interval (int): Number of timesteps between saved models.
    """
    def __init__(self, env, model,
                 max_timesteps=50000,
                 gamma=0.999,
                 batch_size=32,
                 learning_rate=1e-3,
                 exploration_steps=5000,
                 exploration_final_eps=0.05,
                 buffer_size=50000,
                 prioritized_replay_eps=1e-4,
                 prioritized_replay_alpha=0.9,
                 target_network_update_period=2000,
                 distil_dagger_iters=2,
                 distil_epochs=10,
                 dagger_explore_steps=500,
                 log_file=None,
                 save_path=None,
                 save_interval=10000):
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

        self.distil_dagger_iters = distil_dagger_iters
        self.distil_epochs = distil_epochs
        self.dagger_explore_steps = dagger_explore_steps
        self.save_path = save_path
        self.save_interval = save_interval

        self.log_file = log_file

        self.model = model
        self._use_cuda = torch.cuda.is_available()
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=learning_rate)

        if self._use_cuda:
            self.model = self.model.cuda()

    def update_model(self, target_q):
        # sample from replay_buffer
        beta = 0.4 \
            + (1 - 0.40) \
            * self.t / self.max_timesteps
        beta = min(beta, 1)
        replay = self.replay_buffer.sample(self.batch_size, beta=beta)

        # pdb.set_trace()
        # prepare tensors
        tensor_replay = [torch.from_numpy(val) for val in replay[:-1]]
        if self._use_cuda:
            tensor_replay = [val.cuda() for val in tensor_replay]
        states0, actions, rewards, states1, dones, weights = tensor_replay

        # predict target with target network
        max_actions = self.model.forward(Variable(states1.float())).max(-1)[1]
        var_target_action_values = \
            target_q.forward(Variable(states1.float())) \
                    .gather(1, max_actions.unsqueeze(-1)) \
                    .squeeze(-1)
        var_targets = Variable(rewards.float()) \
            + self.gamma * var_target_action_values \
            * (Variable(-dones.float() + 1))
        var_targets = var_targets.detach()

        # gradient descend model
        var_states0 = Variable(states0.float())
        var_action_values = self.model.forward(var_states0) \
            .gather(1, Variable(actions.view(-1, 1))) \
            .squeeze(-1)
        var_loss = (var_action_values - var_targets) ** 2

        # weighted sum loss
        var_weights = Variable(weights.float())
        var_loss_mean = torch.mean(var_loss * var_weights)

        # gradient descend loss
        self.optimizer.zero_grad()
        var_loss_mean.backward()
        self.optimizer.step()

        # update experience priorities
        indices = replay[-1].astype(int)
        loss = torch.abs(var_action_values - var_targets).data
        new_priority = loss + self.prioritized_replay_eps
        new_priority = new_priority.cpu().tolist()
        self.replay_buffer.update_priorities(indices, new_priority)

        return torch.mean(loss)

    def train(self, callbacks=[]):
        # init target network
        target_q = copy.deepcopy(self.model)

        # Get first state
        state0 = self.env.reset()

        # log statics
        loss = 0
        episode_rewards = [0]

        for i in range(self.max_timesteps):
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
                    for callback in callbacks:
                        callback(self, episode_rewards, loss)
                    episode_rewards.append(0)
                else:
                    state0 = state1

            loss = self.update_model(target_q)

            self.t += 1

            # update target network
            if self.t % self.target_network_update_period == 0:
                target_q.load_state_dict(self.model.state_dict())

            if self.t % self.save_interval == 0:
                if self.save_path is not None:
                    self.save('{}-{}'.format(self.save_path, self.t))

    def make_action(self, observation, test=True):
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
            raw = self.get_action_raw(np.expand_dims(observation, 0))
            return np.argmax(raw)

    def get_action_raw(self, observation):
        obs = torch.from_numpy(observation).float()
        var_obs = Variable(obs, volatile=True)
        if self._use_cuda:
            var_obs = var_obs.cuda()
        var_action_value = self.model.forward(var_obs)
        return var_action_value.data.cpu().numpy()

    def learn(self, expert, observations):
        def fit(dataloader, epochs):
            """function that fit self.model with data in dataloader"""
            for epoch in range(epochs):
                for obs, target in dataloader:
                    obs = Variable(obs.float())
                    target = Variable(target.float())
                    if self._use_cuda:
                        obs = obs.cuda()
                        target = target.cuda()
                    predict = self.model.forward(obs)
                    loss = torch.nn.functional.mse_loss(predict, target)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

        # collect target action value on observations
        targets = []
        for i in range(0, observations.shape[0], self.batch_size):
            batch_obs = observations[i:i+self.batch_size]
            target = expert.get_action_raw(batch_obs)
            targets.append(target)  # take care GPU ram usage here
        targets = np.concatenate(targets, axis=0)

        # make dataloader
        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(observations),
            torch.from_numpy(targets))
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True)

        # fit self.model
        fit(dataloader, self.distil_epochs)

        # start doing dagger
        for i in range(self.distil_dagger_iters):
            # collect trajectories explored with self.model
            observations = [observations]
            targets = [targets]
            obs = self.env.reset()
            for step in range(self.dagger_explore_steps):
                observations.append(np.expand_dims(obs, 0))

                # use expert's raw predict as target
                raw = expert.get_action_raw(np.expand_dims(obs, 0))
                targets.append(raw)

                # use self.model to make action
                action = self.make_action(obs)

                # step
                obs, _, done, _ = self.env.step(action)
                if done:
                    obs = self.env.reset()

            # make data loader
            observations = np.concatenate(observations, axis=0)
            targets = np.concatenate(targets, axis=0)
            dataset = torch.utils.data.TensorDataset(
                torch.from_numpy(observations),
                torch.from_numpy(targets))
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True)

            # fit again
            fit(dataloader, self.distil_epochs)

    def get_experience(self):
        return self.replay_buffer.get_experience()

    def save(self, ckp_name, model_only):
        if model_only:
            torch.save({'model': self.model.state_dict()},
                       ckp_name)
        else:
            torch.save({'t': self.t,
                        'model': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'replay_buffer': self.replay_buffer},
                       ckp_name)

    def load(self, ckp_name, model_only=True):
        ckp = torch.load(ckp_name)
        self.model.load_state_dict(ckp['model'])
        if not model_only:
            self.t = ckp['t']
            self.optimizer.load_state_dict(ckp['optimizer'])
            self.replay_buffer = ckp['replay_buffer']

    @staticmethod
    def jointly_make_action(observation, agents, agent_weights):
        raw = AgentDQN.jointly_get_action_raw(
            observation, agents, agent_weights)
        return np.argmax(raw, -1)[0, 0]

    # @staticmethod
    # def jointly_get_action_raw(observation, agents, agent_weights):
    #     # TODO: Parallization
    #     raw = 0
    #     for agent, weight in zip(agents, agent_weights):
    #         raw += agent.get_action_raw(observation) * weight

    #     return raw

    @staticmethod
    def get_fitness(agent1, agent2):
        mean_reward1 = agent1.replay_buffer.get_mean_reward()
        mean_reward2 = agent2.replay_buffer.get_mean_reward()
        return mean_reward1 + mean_reward2
