import copy
import itertools
import numpy as np
from agent_base import AgentBase


class AgentGRL():
    """ A deep Q network agent optimize with genetic algorithm.

    Args:
        base_agent (Agent): Any RL agent.
        agent_clf (Classifier): Any classifier that map observation to
            choice of agent. Used in process of crossover (distillation).
        population_size (int): Population size of GA.
        n_generations (int): Number of generation of GA.
        distil_steps (int): Number of steps to run to do distillation.
        use_cuda (bool): Wheather or not use CUDA (GPU).
    """
    def __init__(self,
                 base_agent,
                 agent_clf,
                 population_size=8,
                 n_generations=100,
                 distil_steps=10000):
        self.base_agent = base_agent
        self.agent_clf = agent_clf
        self.n_generations = n_generations
        self.distil_steps = distil_steps

        # init GA population with random chromosomes
        self.population = [copy.deepcopy(base_agent)
                           for i in range(population_size)]

        # init number of generations
        self.gen = 0

    def train(self):
        """Train it!
        """
        while self.gen < self.n_generations:
            self._mutate()
            parents = self._select()
            children = []
            for parent in parents:
                child = self._crossover(parent)
                children.append(child)
            self.gen += 1

    def _mutate(self):
        """Mutate chromosomes in the population.
        """
        # Todo: parallize it!
        for chrom in self.population:
            chrom.train()

    @staticmethod
    def _evaluate(parent):
        """Evaluate a parent.

        Args:
            (chrom, chrom): 2-tuple of chromosome.
        Returns:
            float: Fitness of the parent.
        """
        pass

    def _select(self):
        """Select parents to generate children.

        Returns:
            [(chrom, chrom)]: List of pair of selected chromosomes.
        """
        parents = list(itertools.combinations(self.population, 2))
        fitnesses = list(map(self._evaluate, parents))
        threshold = sorted(fitnesses)[self.population]
        return list(filter(lambda fitness: fitness >= threshold,
                           fitnesses))

    def _crossover(self, parent):
        """Crossover parent and generate child.

        Returns:
            chrom: Child chromosome.
        """
        # get experience from the parent
        p0_exp = parent[0].get_experience()
        p1_exp = parent[1].get_experience()

        # init classifier
        clf = copy.deepcopy(self.agent_clf)

        # make training data for the classifier
        x = np.concatenate([p0_exp, p1_exp], axis=0)
        y = np.concatenate([np.zeros(p0_exp.shape[0]),
                            np.ones(p1_exp.shape[1])],
                           axis=0)

        # train the classifier
        clf.fit(x, y)

        # build hierarchical agent
        hier_agent = _HierarchicalAgent(clf, parent)

        # init child
        child = copy.deepcopy(self.base_agent)

        # doing distillation
        child.learn(hier_agent)


class _HierarchicalAgent(AgentBase):
    """A hierarchical agent.

    Args:
        clf (classifier): A trained classifier that decides
            which agent to use.
        agents (tuple of Agent): Agents.
    """
    def __init__(self, clf, agents):
        super(_HierarchicalAgent).__init__(self)
        self.clf = clf
        self.agents = agents

    def make_action(self, observation):
        agent_proba = self.clf.predict_proba(observation)
        return type(self.agents[0]) \
            .jointly_make_action(observation, self.agents, agent_proba)

    def get_action_raw(self, observation):
        agent_proba = self.clf.predict_proba(observation)
        return type(self.agents[0]) \
            .jointly_get_action_raw(observation, self.agents, agent_proba)
