import copy
import itertools
import multiprocessing
import os
import sys
import tempfile
import traceback
import numpy as np
from .agent_base import AgentBase
from utils.callbacks import CallbackLog
import pdb


class AgentGRL():
    """ A deep Q network agent optimize with genetic algorithm.

    Args:
        base_agent (Agent): Any RL agent.
        agent_clf (Classifier): Any classifier that map observation to
            choice of agent. Used in process of crossover (distillation).
        population_size (int): Population size of GA.
        n_generations (int): Number of generation of GA.
        use_cuda (bool): Wheather or not use CUDA (GPU).
        n_workers (int): Number of threads to use. If -1, then use the
            number of cpu.
        ckp_dir (str): The directory where checkpoints of agents in
            generations will be saved to. If set to None, agents checkpoints
            will be not saved.
        log_dir (str): The directory where log of generations will be saved
            to. If set to None, log will not be saved.
    """

    def __init__(self,
                 base_agent,
                 agent_clf,
                 population_size=8,
                 n_generations=100,
                 n_workers=1,
                 ckp_dir=None,
                 log_dir=None):
        self.base_agent = base_agent
        self.agent_clf = agent_clf
        self.n_generations = n_generations

        # init GA population with random chromosomes
        self.population = [copy.deepcopy(base_agent)
                           for i in range(population_size)]

        # init number of generations
        self.gen = 0

        self.n_workers = n_workers
        self.ckp_dir = ckp_dir
        self.log_dir = log_dir

    def train(self):
        """Train it!
        """
        while self.gen < self.n_generations:
            print('Generation {}'.format(self.gen))
            self._mutate()
            parents = self._select()
            children = []

            # crossover
            tmp_files = [tempfile.NamedTemporaryFile()
                         for _ in range(len(self.population))]
            ctx = multiprocessing.get_context('spawn')
            with ctx.Pool(self.n_workers) as p:
                for parent, tmp_file in zip(parents, tmp_files):
                    p.apply_async(self._crossover, (parent, tmp_file.name))

                p.close()
                p.join()

            children = [copy.deepcopy(self.base_agent)
                        for _ in range(len(self.population))]
            for child, tmp_file in zip(children, tmp_files):
                child.load(tmp_file.name, model_only=False)
                tmp_file.close()

            self.population = children

            self.gen += 1

    def _mutate(self):
        """Mutate chromosomes in the population.
        """
        # tmp files to store model
        tmp_files = [tempfile.NamedTemporaryFile()
                     for _ in range(len(self.population))]

        if self.log_dir is not None:
            log_files = [os.path.join(self.log_dir,
                                      'gen-%d-%d.log' % (self.gen, i))
                         for i in range(len(self.population))]
        else:
            log_files = [None
                         for i in range(len(self.population))]

        with multiprocessing.get_context('spawn').Pool(self.n_workers) as p:
            for chrom, tmp_file, log_file in zip(self.population,
                                                 tmp_files,
                                                 log_files):
                p.apply_async(_chrom_train_wrapper,
                              (chrom, tmp_file.name, log_file))

            p.close()
            p.join()

        # load model from tmp files
        for chrom, tmp_file in zip(self.population, tmp_files):
            chrom.load(tmp_file.name, model_only=False)
            tmp_file.close()

        # save model if self.chp_dir is specified
        if self.ckp_dir is not None:
            for i, chrom in enumerate(self.population):
                chrom.save(os.path.join(self.ckp_dir,
                                        'gen-%d-%d.ckp' % (self.gen, i)))

    @staticmethod
    def _evaluate(parent):
        """Evaluate a parent.

        Args:
            (chrom, chrom): 2-tuple of chromosome.
        Returns:
            float: Fitness of the parent.
        """
        return parent[0].get_fitness(*parent)

    def _select(self):
        """Select parents to generate children.

        Returns:
            [(chrom, chrom)]: List of pair of selected chromosomes.
        """
        parents = list(itertools.combinations(self.population, 2))
        fitnesses = list(map(self._evaluate, parents))
        threshold = sorted(fitnesses)[-len(self.population)]
        selected = []
        for parent, fitness in zip(parents, fitnesses):
            if fitness >= threshold and len(selected) < len(self.population):
                selected.append(parent)

        assert(len(selected) == len(self.population))
        return selected

    def _crossover(self, parent, ckp_name):
        """Crossover parent and generate child.

        Args:
            parent (Agent, Agent): Two-tuple of agents.
            ckp_name (str): Name of checkpoint to save to.
        """
        # get experience from the parent
        p0_exp = parent[0].get_experience()
        p1_exp = parent[1].get_experience()

        # init classifier
        clf = copy.deepcopy(self.agent_clf)

        # make training data for the classifier
        x = np.concatenate([p0_exp, p1_exp], axis=0)
        y = np.concatenate([np.zeros(p0_exp.shape[0]),
                            np.ones(p1_exp.shape[0])],
                           axis=0)

        # train the classifier
        print('start training classifier')
        clf.fit(x, y)

        # build hierarchical agent
        hier_agent = _HierarchicalAgent(clf, parent)

        # init child
        child = copy.deepcopy(self.base_agent)
        child.set_state(parent[0].get_state())

        # doing distillation
        print('start doing distillation')
        child.learn(hier_agent, x)

        # save result to file ckp_name
        child.save(ckp_name, model_only=False)


class _HierarchicalAgent(AgentBase):
    """A hierarchical agent.

    Args:
        clf (classifier): A trained classifier that decides
            which agent to use.
        agents (tuple of Agent): Agents.
    """

    def __init__(self, clf, agents):
        super(_HierarchicalAgent, self).__init__()
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


def _chrom_train_wrapper(chrom, ckp_name, log_file):
    try:
        if log_file is not None:
            callbacks = [CallbackLog(log_file).on_episode_end]
        else:
            callbacks = []
        chrom.train(callbacks)
        chrom.save(ckp_name, model_only=False)
    except BaseException as e:
        print('Error occured in subproces!', file=sys.stderr)
        traceback.print_exc()
