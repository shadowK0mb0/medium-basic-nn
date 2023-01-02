import argparse
import torch

from torch.utils.data import DataLoader

import data
import model
import trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="data/FashionMNIST/raw/")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--hidden-size", type=int, default=512)
    args = parser.parse_args() 
    
    X_train, y_train = data.load_mnist(args.data_path, kind='train')
    X_test, y_test = data.load_mnist(args.data_path, kind='t10k')
    
    training_data = data.CustomFashinMNSITDataset(torch.Tensor(X_train), torch.LongTensor(y_train))
    test_data = data.CustomFashinMNSITDataset(torch.Tensor(X_test), torch.LongTensor(y_test))   

    train_loader = DataLoader(training_data, batch_size=args.batch_size)
    test_loader = DataLoader(test_data, batch_size=args.batch_size)
    
    model = model.FashionMnistClassifier(args.hidden_size)
    
    loss_fn = torch.nn.CrossEntropyLoss()
    lr = args.lr
    epochs = args.epochs
    
    fashionMNIST_trainer = trainer.Trainer(train_loader, test_loader, model, lr, loss_fn)
    fashionMNIST_trainer.train(epochs)