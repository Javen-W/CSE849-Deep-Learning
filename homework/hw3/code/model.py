import torch
import torch.nn as nn

torch.manual_seed(123)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm2d(num_features=16)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=32)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=48, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=48)

        self.conv4 = nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(num_features=64)

        self.conv5 = nn.Conv2d(in_channels=64, out_channels=80, kernel_size=3, padding=1)

        self.fc = nn.Linear(in_features=80, out_features=10)

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=80)

    def forward(self, x, intermediate_outputs=False):
        # TODO: Compute the forward pass output following the diagram in
        # the project PDF. If intermediate_outputs is True, return the
        # outputs of the convolutional layers as well.

        conv1_out = self.conv1(x)
        x = self.relu(self.bn1(conv1_out))

        conv2_out = self.conv2(x)
        x = self.maxpool(self.relu(self.bn2(conv2_out)))

        conv3_out = self.conv3(x)
        x = self.maxpool(self.relu(self.bn3(conv3_out)))

        conv4_out = self.conv4(x)
        x = self.maxpool(self.relu(self.bn4(conv4_out)))

        conv5_out = self.conv5(x)
        x = self.fc(self.avgpool(conv5_out))

        final_out = x

        if intermediate_outputs:
            return final_out, [conv1_out, conv2_out, conv3_out, conv4_out, conv5_out]
        else:
            return final_out
