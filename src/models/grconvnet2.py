import torch.nn as nn
import torch.nn.functional as F

from src.models.grasp_model import ResidualBlock
from src.train_testers import GraspTrainTester


class GenerativeResnet2(nn.Module):

	def __init__(self, input_channels=4, output_channels=1, channel_size=32, dropout=False, prob=0.0):
		super(GenerativeResnet2, self).__init__()
		self.conv1 = nn.Conv2d(input_channels, channel_size, kernel_size=9, stride=1, padding=4)
		self.bn1 = nn.BatchNorm2d(channel_size)

		self.conv2 = nn.Conv2d(channel_size, channel_size * 2, kernel_size=4, stride=2, padding=1)
		self.bn2 = nn.BatchNorm2d(channel_size * 2)

		self.conv3 = nn.Conv2d(channel_size * 2, channel_size * 4, kernel_size=4, stride=2, padding=1)
		self.bn3 = nn.BatchNorm2d(channel_size * 4)

		self.res1 = ResidualBlock(channel_size * 4, channel_size * 4, dropout=dropout, prob=prob)
		self.res2 = ResidualBlock(channel_size * 4, channel_size * 4, dropout=dropout, prob=prob)
		self.res3 = ResidualBlock(channel_size * 4, channel_size * 4, dropout=dropout, prob=prob)
		self.res4 = ResidualBlock(channel_size * 4, channel_size * 4, dropout=dropout, prob=prob)
		self.res5 = ResidualBlock(channel_size * 4, channel_size * 4, dropout=dropout, prob=prob)

		self.conv4 = nn.ConvTranspose2d(channel_size * 4, channel_size * 2, kernel_size=4, stride=2, padding=1,
										output_padding=1)
		self.bn4 = nn.BatchNorm2d(channel_size * 2)

		self.conv5 = nn.ConvTranspose2d(channel_size * 2, channel_size, kernel_size=4, stride=2, padding=2,
										output_padding=1)
		self.bn5 = nn.BatchNorm2d(channel_size)

		self.conv6 = nn.ConvTranspose2d(channel_size, channel_size, kernel_size=9, stride=1, padding=4)

		self.pos_decoder = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
		self.cos_decoder = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
		self.sin_decoder = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
		self.width_decoder = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
		self.graspness_decoder = nn.Conv2d(in_channels=channel_size, out_channels=1, kernel_size=2)
		self.bin_classifier = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)

		self.dropout = dropout
		self.dropout_pos = nn.Dropout(p=prob)
		self.dropout_cos = nn.Dropout(p=prob)
		self.dropout_sin = nn.Dropout(p=prob)
		self.dropout_wid = nn.Dropout(p=prob)

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

		if self.dropout:
			pos_output = self.pos_decoder(self.dropout_pos(x))
			cos_output = self.cos_decoder(self.dropout_cos(x))
			sin_output = self.sin_decoder(self.dropout_sin(x))
			width_output = self.width_decoder(self.dropout_wid(x))
		else:
			pos_output = self.pos_decoder(x)
			cos_output = self.cos_decoder(x)
			sin_output = self.sin_decoder(x)
			width_output = self.width_decoder(x)

		graspness_output = self.graspness_decoder(x)
		bin_output = self.bin_classifier(x)

		return (
			pos_output.squeeze(1),
			cos_output.squeeze(1),
			sin_output.squeeze(1),
			width_output.squeeze(1),
			graspness_output.squeeze(1),
			bin_output.squeeze(1),
		)

def train_test(config, model_params={}):
	"""Train and test a net."""
	net = GenerativeResnet2(4 if config.use_rgbd_img else 1, config.num_of_bins, dropout=config.use_dropout, prob=config.dropout_prob)
	features = {'depth_images', 'grasp_targets', 'grasps'}
	train_tester = GraspTrainTester(net, config, features)
	train_tester.train_test()