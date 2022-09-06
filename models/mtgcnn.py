import torch.nn as nn
import torch.nn.functional as F

filter_sizes = [32, 16, 8, 8, 16, 32]
kernel_sizes = [9, 5, 3, 3, 5, 9]
strides = [3, 2, 2, 2, 2, 3]


class MTGCNN(nn.Module):
	"""
	GG-CNN
	Equivalient to the Keras Model used in the RSS Paper (https://arxiv.org/abs/1804.05172)
	"""
	def __init__(self, input_channels=1):
		super().__init__()
		self.conv1 = nn.Conv2d(input_channels, filter_sizes[0], kernel_sizes[0], stride=strides[0], padding=3)
		self.conv2 = nn.Conv2d(filter_sizes[0], filter_sizes[1], kernel_sizes[1], stride=strides[1], padding=2)
		self.conv3 = nn.Conv2d(filter_sizes[1], filter_sizes[2], kernel_sizes[2], stride=strides[2], padding=1)
		self.convt1 = nn.ConvTranspose2d(filter_sizes[2], filter_sizes[3], kernel_sizes[3], stride=strides[3], padding=1, output_padding=1)
		self.convt2 = nn.ConvTranspose2d(filter_sizes[3], filter_sizes[4], kernel_sizes[4], stride=strides[4], padding=2, output_padding=1)
		self.convt3 = nn.ConvTranspose2d(filter_sizes[4], filter_sizes[5], kernel_sizes[5], stride=strides[5], padding=3, output_padding=1)

		self.convt4 = nn.ConvTranspose2d(filter_sizes[2], filter_sizes[3], kernel_sizes[3], stride=strides[3], padding=1, output_padding=1)
		self.convt5 = nn.ConvTranspose2d(filter_sizes[3], filter_sizes[4], kernel_sizes[4], stride=strides[4], padding=2, output_padding=1)
		self.convt6 = nn.ConvTranspose2d(filter_sizes[4], filter_sizes[5], kernel_sizes[5], stride=strides[5], padding=3, output_padding=1)

		self.pos_output = nn.Conv2d(filter_sizes[5], 1, kernel_size=2)
		self.cos_output = nn.Conv2d(filter_sizes[5], 1, kernel_size=2)
		self.sin_output = nn.Conv2d(filter_sizes[5], 1, kernel_size=2)
		self.width_output = nn.Conv2d(filter_sizes[5], 1, kernel_size=2)
		self.depth_output = nn.Conv2d(filter_sizes[5], 1, kernel_size=2)

		for m in self.modules():
			if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
				nn.init.xavier_uniform_(m.weight, gain=1)

	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = F.relu(self.conv3(x))
		y = F.relu(self.convt1(x))
		y = F.relu(self.convt2(y))
		y = F.relu(self.convt3(y))

		z = F.relu(self.convt4(x))
		z = F.relu(self.convt5(z))
		z = F.relu(self.convt6(z))

		pos_output = self.pos_output(y)
		cos_output = self.cos_output(y)
		sin_output = self.sin_output(y)
		width_output = self.width_output(y)

		depth_output = self.depth_output(z)

		return pos_output, cos_output, sin_output, width_output, depth_output

	def compute_loss(self, xc, yc, newloss=False):
		y_pos, y_cos, y_sin, y_width, y_depth = yc
		pos_pred, cos_pred, sin_pred, width_pred, depth_pred = self(xc)

		p_loss = F.mse_loss(pos_pred, y_pos)
		if newloss == False:
			cos_loss = F.mse_loss(cos_pred, y_cos)
			sin_loss = F.mse_loss(sin_pred, y_sin)
			width_loss = F.mse_loss(width_pred, y_width)
		else:
			cos_loss = F.mse_loss(torch.mul(y_pos, cos_pred), torch.mul(y_pos, y_cos))
			sin_loss = F.mse_loss(torch.mul(y_pos, sin_pred), torch.mul(y_pos, y_sin))
			width_loss = F.mse_loss(torch.mul(y_pos, width_pred), torch.mul(y_pos, y_width))		

		depth_loss = F.mse_loss(depth_pred, y_depth)

		return {
			'loss': p_loss + cos_loss + sin_loss + width_loss + depth_loss,
			'losses': {
				'p_loss': p_loss,
				'cos_loss': cos_loss,
				'sin_loss': sin_loss,
				'width_loss': width_loss,
				'depth_loss': depth_loss
			},
			'pred': {
				'pos': pos_pred,
				'cos': cos_pred,
				'sin': sin_pred,
				'width': width_pred,
				'depth': depth_pred
			}
		}
