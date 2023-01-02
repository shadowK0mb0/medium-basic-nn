import typing

import torch

class Trainer:
    def __init__(self, 
                 train_dataset: torch.utils.dataset,
                 test_dataset: torch.utils.dataset,
                 batch_size: int, 
                 model: torch.nn.module,
                 lr: float,
                 loss_fn: typing.Callable,
                 optimizer: torch.optim.Optimizer) -> None:
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
        self.model = model
        self.lr = lr
        self.loss_fn = loss_fn
        self.optimizer = optimizer
    
    def train(self, epochs: int) -> None:
        train_losses = []
        test_losses = []
        for i in range(epochs):
            print(f"Epoch {i+1}\n-------------------------------")
            train_losses.append(self.train_one_epoch())
            test_losses.append(self.test())
            
    def train_one_epoch(self) -> float:
        size = len(self.train_loader.dataset)
        loss = 0
        for batch, (X, y) in enumerate(self.train_loader):
            # Compute prediction and loss
            pred = self.model(X)
            loss = self.loss_fn(pred, y)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
 
            losses += loss.item()

            if batch % 100 == 0:
                loss, current = losses[-1], batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        return losses/len(self.train_loader)
    
    def test(self) -> float:
        size = len(self.test_loader.dataset)
        num_batches = len(self.test_loader)
        test_loss, correct = 0, 0

        with torch.no_grad():
            for X, y in self.test_loader:
                pred = self.model(X)
                test_loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        return test_loss