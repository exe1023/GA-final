import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

# Factorised NoisyLinear layer with bias
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.4):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer('bias_epsilon', torch.Tensor(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.weight_mu.size(1))
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))

    def _scale_noise(self, size):
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.out_features))

    def forward(self, input):
        if self.training:
            return F.linear(input, self.weight_mu + self.weight_sigma.mul(Variable(self.weight_epsilon)), self.bias_mu + self.bias_sigma.mul(Variable(self.bias_epsilon)))
        else:
            return F.linear(input, self.weight_mu, self.bias_mu)


class DQN(nn.Module):
    def __init__(self, channels, 
                 num_actions,
                 dueling=False,
                 noise_linear=False):
        super(DQN, self).__init__()
        self.dueling = dueling
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        #self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        linear = nn.Linear
        if noise_linear:
            linear = NoisyLinear
            
        self.fc = linear(3136, 512)

        if self.dueling:
            self.fc_v = linear(512, 1)
            self.fc_As = linear(512, num_actions)
        else:
            self.fc_head = linear(512, num_actions)
        
        self.relu = nn.ReLU()
        self.lrelu = nn.LeakyReLU(0.01)


    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        #x = self.relu(self.conv4(x))
        x = self.lrelu(self.fc(x.view(x.size(0), -1)))
        if self.dueling:
            v = self.fc_v(x)
            As = self.fc_As(x)
            q = v.expand_as(As) + (As - As.mean(1, keepdim=True).expand_as(As))
        else:
            q = self.fc_head(x)
        return q

    def reset_noise(self):
        for name, module in self.named_children():
            if 'fc' in name:
                module.reset_noise()

if __name__ == '__main__':
    # test forward function
    dummy = torch.Tensor(32, 4, 84, 84)
    dummy = torch.autograd.Variable(dummy)
    dqn = DQN(4, 4)
    dqn(dummy)

    