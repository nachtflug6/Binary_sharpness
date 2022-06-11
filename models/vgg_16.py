import torch as th
import torch.nn as nn
import torchvision as tv
import torch.nn.functional as tf


class BinaryVgg16(nn.Module):
    def __init__(self, params):
        super(BinaryVgg16, self).__init__()

        standard_vgg16 = tv.models.vgg16(pretrained=False)
        sequentials = list(standard_vgg16.children())
        self.feature_extractor = sequentials[0]
        self.flattening = sequentials[1]
        self.fully_connected = sequentials[2]

        # Change input layer to 1 color channel
        self.feature_extractor[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # Change output layer to 1 value
        self.fully_connected[-1] = nn.Linear(4096, 1, bias=True)

    def forward(self, input_tensor):

        output = self.feature_extractor(input_tensor)
        output = self.flattening(output)
        output = th.flatten(output, start_dim=1, end_dim=3)
        output = self.fully_connected(output)
        output = th.sigmoid(output)

        return output
