import numpy as np
import torch as th
from tqdm import tqdm
import torch.nn as nn


class BinaryTrainer:
    def __init__(self, model: th.nn.Module, device: th.device, criterion: th.nn.Module, optimizer: th.optim.Optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        # self.ds_train = ds_train
        # self.ds_test = ds_test
        self.epochs = 0
        self.train_losses = []
        self.test_losses = []

    def train_epoch(self, trainloader):

        print('Epoch: ', self.epochs, ' ----------------------')

        self.model.train()
        current_loss = []
        self.epochs += 1

        loop = tqdm(trainloader)
        random_img = random_target = random_pred = None

        for j, data in enumerate(loop, 0):

            img, target = data

            target = target.to(self.device)
            img = img.to(self.device)
            x_predicted = self.model.forward(img)

            if j == 0:
                random_img = img.cpu().detach().numpy()
                random_target = target.cpu().detach().numpy()
                random_pred = x_predicted.cpu().detach().numpy()

            loss = self.criterion(x_predicted, target)

            self.optimizer.zero_grad()
            loss.backward()  # th.ones_like(loss))
            self.optimizer.step()

            current_loss.append(th.mean(loss).item())
            #print('pred: ', x_predicted, 'target: ', target)
            #print(loss.item())

            loop.set_description(f"Train")
            #loop.set_postfix(target=target.item(), pred=x_predicted.item())
            loop.set_postfix(loss=np.mean(current_loss))

        self.train_losses.append(np.mean(current_loss))

        return random_img, random_target, random_pred

    def test(self, testloader):
        self.model.eval()
        current_loss = []

        loop = tqdm(testloader)


        random_img = random_target = random_pred = None

        for j, data in enumerate(loop, 0):
            img, target = data
            target = target.to(self.device)
            img = img.to(self.device)

            x_predicted = self.model.forward(img)

            if j == 0:
                random_img = img.cpu().detach().numpy()
                random_target = target.cpu().detach().numpy()
                random_pred = x_predicted.cpu().detach().numpy()

            x_predicted = th.where(x_predicted > 0.5, 1.0, 0.0)

            loss = nn.functional.l1_loss(x_predicted, target)

            current_loss.append(th.mean(loss).item())

            loop.set_description(f"Test")
            loop.set_postfix(acc=1 - np.mean(current_loss))

        self.test_losses.append(np.mean(current_loss))

        return random_img, random_target, random_pred