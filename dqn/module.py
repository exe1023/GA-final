import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, channels, num_actions, dueling=False):
        super(DQN, self).__init__()
        self.dueling = dueling
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        #self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc = nn.Linear(3136, 512)
        if self.dueling:
            self.v = nn.Linear(512, 1)
            self.As = nn.Linear(512, num_actions)
        else:
            self.head = nn.Linear(512, num_actions)
        
        self.relu = nn.ReLU()
        self.lrelu = nn.LeakyReLU(0.01)


    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        #x = self.relu(self.conv4(x))
        x = self.lrelu(self.fc(x.view(x.size(0), -1)))
        if self.dueling:
            v = self.v(x)
            As = self.As(x)
            q = v.expand_as(As) + (As - As.mean(1, keepdim=True).expand_as(As))
        else:
            q = self.head(x)
        return q

if __name__ == '__main__':
    # test forward function
    dummy = torch.Tensor(32, 4, 84, 84)
    dummy = torch.autograd.Variable(dummy)
    dqn = DQN(4, 4)
    dqn(dummy)

    