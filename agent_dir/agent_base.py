import numpy as np


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
            np array: Raw values for each action.
        """
        pass

    def learn(self, expert, experience):
        """Distil knowledge from agent.

        Args:
            expert (AgentBase): Agent to learn from.
            experience (np array): Observation the agent experienced.
        """
        pass

    def get_experience(self, agent):
        """Return observations the agent experienced.

        Returns:
            np array: Observation the agent experienced.
        """
        pass

    @staticmethod
    def jointly_make_action(self, observation, agents, agent_weights):
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
    def jointly_get_action_raw(self, observation, agents, agent_weights):
        """Return action value according to the agents jointly.

        Args:
            observation (np array): Observation.
            agents ([AgentBase]): List of agents.
            agent_weights (np array): np array of shape (n_agents,) that
                represents weights of each agent.
        Returns:
            np array: np array of shape (n_actions,).
        """
        values = [agent.get_action_raw(observation)
                  for agent in agents]
        values = np.array(values)
        return (agent_weights.reshape(1, -1) @ values.T).reshape(-1)
