

class AgentBase:
    """A base virtual class for agent that are used as base agent in AgentGRL.
    """

    def __init__(self):
        pass

    def train(self):
        """Train!!
        """

    def make_action(self, observation):
        """Make action given observation.

        Args:
            observation (np.array): Observation.
        Returns:
            int: Action to take.
        """
        pass

    def get_action_raw(self, observation):
        """Return raw value of actions.

        Args:
            observation (np.array): Observation.
        Returns:
            np array: Raw values for each action with shape
                (batch_size, n_actions).
        """
        pass

    def learn(self, expert, experience):
        """Distil knowledge from agent.

        Args:
            expert (AgentBase): Agent to learn from.
            experience (np array): Observation the agent experienced.
        """
        pass

    def get_experience(self):
        """Return observations the agent experienced.

        Returns:
            np array: Observation the agent experienced.
        """
        pass

    def save(self, ckp_name, model_only=True):
        """Save the model and some information like replay buffer to file
        `ckp_name`.

        Args:
            ckp_name (str): Name of the saved file.
            model_only (bool): Whether or not only save model, but any other
                information.
        """
        pass

    def load(self, ckp_name, model_only=True):
        """Load the model and some information like replay buffer.

        Args:
            ckp_name (str): Name of the file to load.
            model_only (bool): Whether or not only load model, but any other
                information.
        """
        pass

    @staticmethod
    def jointly_make_action(observation, agents, agent_weights):
        """Make action according to agents the jointly.

        This method should mix opinions from agents. For example, if the agents
        are critic based, then it should returns action according to weighted
        mean of the value predicted by the agents. If the agents are actor
        based, then it should sample action from weighted mean of the
        distribution predicted by the agents.

        Args:
            observation (np array): Observation.
            agents ([AgentBase]): List of agents.
            agent_weights (np array): np array of shape (n_agents,) that
                represents weights of each agent.
        Returns:
            int: Action to take.
        """
        pass

    @staticmethod
    def jointly_get_action_raw(observation, agents, agent_weights):
        """Return action value according to the agents jointly.

        Args:
            observation (np array): Observation.
            agents ([AgentBase]): List of agents.
            agent_weights (np array): np array of shape (n_agents,) that
                represents weights of each agent.
        Returns:
            np array: np array of shape (batch_size, n_actions).
        """
        action_raw = 0
        for agent, weight in zip(agents, agent_weights.T):
            action_raw += weight.reshape(-1, 1) \
                          * agent.get_action_raw(observation)
        return action_raw

    @staticmethod
    def get_fitness(agent1, agent2):
        """Return fitness of pair of agents.

        Args:
            agent1 (AgentBase): First agent in the pair.
            agent2 (AgentBase): Second agent in the pair.

        Returns:
            float: fitness of the pair of agents.
        """
        pass
