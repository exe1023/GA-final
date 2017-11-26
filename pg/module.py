import torch
import torch.nn as nn
import torch.nn.functional as F

class REINFORCE(nn.Module):
    def __init__(self, input_size, num_actions):
        super(REINFORCE, self).__init__()
        self.fc = nn.Linear(input_size, 256)
        self.head = nn.Linear(256, num_actions)
        
        self.relu = nn.ReLU()
        self.elu = nn.ELU()


    def forward(self, x):
        x = self.elu(self.fc(x.view(x.size(0), -1)))
        x = self.head(x)
        return F.softmax(x)

if __name__ == '__main__':
    # test forward function
    dummy = torch.Tensor(32, 4, 84, 84)
    dummy = torch.autograd.Variable(dummy)
    reinforce = REINFORCE(4, 4)
    reinforce(dummy)

    