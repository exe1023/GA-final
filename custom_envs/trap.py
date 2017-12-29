import random

class Space:
    '''
    Some hack to make our env compatible with openai gym
    '''
    def __init__(self, n, shape):
        self.shape = shape
        self.n = n

class MKTrap:
    def __init__(self, m=20, k=5):
        self.problem_size = m * k
        self.m = 20
        self.k = 5
        
        self.observation_space = Space(n=None, shape=(self.problem_size, ))
        self.action_space = Space(n=2, shape=None)
        print('Best Reward:', 2 * m)

    
    def reset(self):
        self.traps = random.sample(range(self.problem_size), 
                                   self.problem_size)
        self.state = [-1 for _ in range(self.problem_size)]
        self.idx = 0
        self.done = False
        return self.state

    def step(self, action):
        assert action in [0, 1]
        self.state[self.idx] = action
        self.idx += 1
        
        self.done = (self.idx >= self.problem_size)
        reward = self.reward()
        info = None

        return self.state, self.reward(), self.done, info

    def seed(self, seed):
        return seed 

    def reward(self):
        '''
        The best BB is 111...1 and its reward is 2
        The deceptive BB is 000...0 and its reward is 1
        Reward of other BBs is (# of 0) * (1/k)
        '''
        if not self.done:
            return 0
        
        reward = 0
        for i in range(0, self.problem_size, self.k):
            trap = self.traps[i: i + self.k]
            num_0 = 0
            for t in trap:
                num_0 += 1 if self.state[t] == 0 else 0
            reward += 2 if num_0 == 0 else num_0 * (1/self.k)
        return reward