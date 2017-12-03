import math
import torch


def _calc_shape(shape, kernel_size, stride, padding, dilation=1):
    shape = list(shape)
    shape[0] = math.floor((shape[0] + 2 * padding
                           - dilation * (kernel_size - 1)
                           - 1)
                          / stride + 1)
    shape[1] = math.floor((shape[1] + 2 * padding
                           - dilation * (kernel_size - 1)
                           - 1)
                          / stride + 1)
    return shape


class CNNClassifier(torch.nn.Module):
    def __init__(self, input_shape, n_classes):
        super(CNNClassifier, self).__init__()
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(input_shape[-1], 8, 5,
                            stride=2, padding=2),
            torch.nn.ELU(),
            torch.nn.Conv2d(8, 16, 5,
                            stride=2, padding=2),
            torch.nn.ELU(),
            torch.nn.Conv2d(16, 16, 3,
                            stride=2, padding=1),
            torch.nn.ELU(),
        )

        shape = input_shape[-3:-1]
        shape = _calc_shape(shape, 5, 2, 2)
        shape = _calc_shape(shape, 5, 2, 2)
        shape = _calc_shape(shape, 3, 2, 1)

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(shape[0] * shape[1] * 16, n_classes)
        )

    def forward(self, observation):
        observation = observation.transpose(-3, -1)
        logits = self.cnn.forward(observation)
        action = self.mlp.forward(logits.view(logits.size(0), -1))
        return action


class DNNClassifier(torch.nn.Module):
    def __init__(self, input_shape, n_classes):
        super(DNNClassifier, self).__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_shape[0], 512),
            torch.nn.ELU(),
            torch.nn.Linear(512, n_classes),
            torch.nn.Softmax()
        )

    def forward(self, observation):
        return self.mlp(observation)
