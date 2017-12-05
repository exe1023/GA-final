import torch


class Classifier(torch.nn.Module):
    def __init__(self, network,
                 batch_size=32,
                 learning_rate=1e-3,
                 max_iters=1000):
        super(Classifier, self).__init__()
        self.network = network
        self.optimizer = torch.optim.Adam(self.network.parameters(),
                                          lr=learning_rate)
        self.loss = torch.nn.CrossEntropyLoss()
        self.max_iters = max_iters
        self.n_epochs = 0

    def fit(self, x, y):
        data = {'x': x, 'y': y}
        dataloader = torch.utils.data.DataLoader(
            data,
            batch_size=self.batch_size,
            shuffle=True)

        while self.n_epochs < self.max_epochs:
            for batch in dataloader:
                y_ = self.network.forward(batch['x'])
                loss = self.loss(y_, batch['y'])

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.n_epochs += 1
