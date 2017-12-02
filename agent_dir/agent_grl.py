import copy
import itertools
import torch


class AgentGRL():
    """ A deep Q network agent optimize with genetic algorithm.

    Args:
        env (gym): Env made with gym.
        base_agent (Agent): Any RL agent.
        agent_clf (Classifier): Any classifier that map observation to
            choice of agent. Used in process of crossover (distillation).
        population_size (int): Population size of GA.
        n_generations (int): Number of generation of GA.
        distil_steps (int): Number of steps to run to do distillation.
        use_cuda (bool): Wheather or not use CUDA (GPU).
    """
    def __init__(self, env,
                 base_agent,
                 agent_clf,
                 population_size=8,
                 n_generations=100,
                 distil_steps=10000,
                 use_cuda=None):
        self.env = env
        self.agent_clf = agent_clf
        self.n_generations = n_generations
        self.distil_steps = distil_steps
        if use_cuda is None:
            use_cuda = torch.cuda.is_available()
        self.use_cuda = use_cuda

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
            chrom.steps = 0
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
        clfs = []
        pass
    
