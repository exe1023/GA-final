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

class REINFORCE_ATARI(nn.Module):
    def __init__(self, channels, num_actions):
        super(REINFORCE_ATARI, self).__init__()
        self.conv1 = nn.Conv2d(channels, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        #self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        #self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc = nn.Linear(32 * 8 * 8, 128)
        self.head = nn.Linear(128, num_actions)
        
        self.relu = nn.ReLU()
        self.lrelu = nn.LeakyReLU(0.01)


    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        #x = self.relu(self.conv3(x))
        #x = self.relu(self.conv4(x))
        x = self.relu(self.fc(x.view(x.size(0), -1)))
        x = self.head(x)
        return F.softmax(x)

if __name__ == '__main__':
    # test forward function
    dummy = torch.Tensor(32, 4, 84, 84)
    dummy = torch.autograd.Variable(dummy)
    reinforce = REINFORCE(4, 4)
    reinforce(dummy)

    