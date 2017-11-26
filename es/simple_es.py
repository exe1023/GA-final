'''
Wrapped version of https://gist.github.com/flyman3046/d37680eeaac469a4030c690ae65b0419
'''
import numpy as np

class ES(object):
    def __init__(self, 
                 num_params,
                 npop=50,
                 sigma=0.1,
                 alpha=0.001):
        '''
        Args:
            num_params: number of model parameters
            npop: population size
            sigma: noise standard deviation
            alpha: learning rate
        '''
        self.num_params = num_params
        self.npop = npop
        self.sigma = sigma
        self.alpha = alpha

        self.solution = np.random.rand(num_params)

    def ask(self):
        '''
        Returns a list of parameters with guassian noise
        '''
        # samples from a normal distribution N(0,1)
        self.N = np.random.randn(self.npop, self.num_params) 
        
        solutions = []
        for i in range(self.npop):
            # jitter w using gaussian of sigma
            solutions.append(self.solution + self.sigma * self.N[i])
        
        return solutions
    
    def tell(self, rewards):
        '''
        Args:
            rewards: np.array, shape = (npop)
        '''
        # standardize the rewards to have a gaussian distribution
        A = (rewards - np.mean(rewards)) / np.std(rewards)
        # perform the parameter update. The matrix multiply below
        # is just an efficient way to sum up all the rows of the noise matrix N,
        # where each row N[j] is weighted by A[j]
        self.solution = self.solution + self.alpha / (self.npop * self.sigma) * np.dot(self.N.T, A)
        