import torch
import torch.utils.data
from torch.autograd import Variable


class Classifier(torch.nn.Module):
    def __init__(self, network,
                 batch_size=32,
                 learning_rate=1e-3,
                 max_epochs=1000):
        super(Classifier, self).__init__()
        self.network = network
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.optimizer = torch.optim.Adam(self.network.parameters(),
                                          lr=learning_rate)
        self.loss = torch.nn.CrossEntropyLoss()
        self.n_epochs = 0

    def fit(self, x, y):
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).long()
        dataset = torch.utils.data.TensorDataset(x, y)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True)

        while self.n_epochs < self.max_epochs:
            for batch in dataloader:
                x, y = batch
                x, y = Variable(x), Variable(y)
                y_ = self.network.forward(x)
                loss = self.loss(y_, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.n_epochs += 1

    def predict_proba(self, x):
        var_x = Variable(torch.from_numpy(x).float())
        return self.network.forward(var_x).data.cpu().numpy()
