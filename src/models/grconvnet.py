import torch.nn as nn
import torch.nn.functional as F

from src.models.grasp_model import ResidualBlock
from src.train_testers import GraspTrainTester


class GenerativeResnet(nn.Module):

    def __init__(self, input_channels=1, output_channels=1, dropout=False, prob=0.0, channel_size=32):
        super(GenerativeResnet, self).__init__()
        self.num_outs = output_channels
        
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=9, stride=1, padding=4)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.res1 = ResidualBlock(128, 128)
        self.res2 = ResidualBlock(128, 128)
        self.res3 = ResidualBlock(128, 128)
        self.res4 = ResidualBlock(128, 128)
        self.res5 = ResidualBlock(128, 128)

        self.conv4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        self.conv5 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=2, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)

        self.conv6 = nn.ConvTranspose2d(32, 32, kernel_size=9, stride=1, padding=4)

        self.pos_decoder = nn.Conv2d(32, output_channels, kernel_size=2)
        self.cos_decoder = nn.Conv2d(32, output_channels, kernel_size=2)
        self.sin_decoder = nn.Conv2d(32, output_channels, kernel_size=2)
        self.width_decoder = nn.Conv2d(32, output_channels, kernel_size=2)
        self.graspness_decoder = nn.Conv2d(32, 1, kernel_size=2)
        self.bin_classifier = nn.Conv2d(32, output_channels, kernel_size=2)

        self.dropout1 = nn.Dropout(p=prob)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x_in):
        x = F.relu(self.bn1(self.conv1(x_in)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.conv6(x)

        return (
            self.pos_decoder(self.dropout1(x)).squeeze(1),
            self.cos_decoder(self.dropout1(x)).squeeze(1),
            self.sin_decoder(self.dropout1(x)).squeeze(1),
            self.width_decoder(self.dropout1(x)).squeeze(1),
            self.graspness_decoder(self.dropout1(x)).squeeze(1),
            self.bin_classifier(self.dropout1(x)).squeeze(1)
        )

def train_test(config, model_params={}):
    """Train and test a net."""
    net = GenerativeResnet(4 if config.use_rgbd_img else 1, config.num_of_bins, dropout=config.use_dropout, prob=config.dropout_prob)
    features = {'depth_images', 'grasp_targets', 'grasps'}
    train_tester = GraspTrainTester(net, config, features)
    train_tester.train_test()