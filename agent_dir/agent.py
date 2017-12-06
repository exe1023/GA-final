from environment import Environment

class Agent(object):
    def __init__(self,args):
        self.env = Environment(args)


    def make_action(self, observation, test=True):
        """
        This function must exist in agent.        

        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (80, 80, 4) 
        
        Return:
            action: int
                the predicted action from trained model
        """
        try:
            return action
        except:
            raise NotImplementedError("Subclasses should implement this!")


    def init_game_setting(self):
        """
        
        testing function will call this function at the begining of new game
        put anything you want to initialize if necessary 
        
        """
        pass