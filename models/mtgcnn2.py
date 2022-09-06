import torch
import torch.nn as nn
import torch.nn.functional as F


class MTGCNN2(nn.Module):
	def __init__(self, input_channels=1, filter_sizes=None, l3_k_size=5, dilations=None):
		super().__init__()

		if filter_sizes is None:
			filter_sizes = [16,  # First set of convs
							16,  # Second set of convs
							32,  # Dilated convs
							16]  # Transpose Convs

		if dilations is None:
			dilations = [2, 4]

		self.features = nn.Sequential(
			# 4 conv layers.
			nn.Conv2d(input_channels, filter_sizes[0], kernel_size=11, stride=1, padding=5, bias=True),
			nn.ReLU(inplace=True),
			nn.Conv2d(filter_sizes[0], filter_sizes[0], kernel_size=5, stride=1, padding=2, bias=True),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2),

			nn.Conv2d(filter_sizes[0], filter_sizes[1], kernel_size=5, stride=1, padding=2, bias=True),
			nn.ReLU(inplace=True),
			nn.Conv2d(filter_sizes[1], filter_sizes[1], kernel_size=5, stride=1, padding=2, bias=True),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2),
		)

		self.grasp = nn.Sequential(
			# Dilated convolutions.
			nn.Conv2d(filter_sizes[1], filter_sizes[2], kernel_size=l3_k_size, dilation=dilations[0], stride=1, padding=(l3_k_size//2 * dilations[0]), bias=True),
			nn.ReLU(inplace=True),
			nn.Conv2d(filter_sizes[2], filter_sizes[2], kernel_size=l3_k_size, dilation=dilations[1], stride=1, padding=(l3_k_size//2 * dilations[1]), bias=True),
			nn.ReLU(inplace=True),

			# Output layers
			nn.ConvTranspose2d(filter_sizes[2], filter_sizes[3], 3, stride=2, padding=1, output_padding=1),
			nn.ReLU(inplace=True),
			nn.ConvTranspose2d(filter_sizes[3], filter_sizes[3], 3, stride=2, padding=1, output_padding=1),
			nn.ReLU(inplace=True),
		)

		self.depth = nn.Sequential(
			# Dilated convolutions.
			nn.Conv2d(filter_sizes[1], filter_sizes[2], kernel_size=l3_k_size, dilation=dilations[0], stride=1, padding=(l3_k_size//2 * dilations[0]), bias=True),
			nn.ReLU(inplace=True),
			nn.Conv2d(filter_sizes[2], filter_sizes[2], kernel_size=l3_k_size, dilation=dilations[1], stride=1, padding=(l3_k_size//2 * dilations[1]), bias=True),
			nn.ReLU(inplace=True),

			# Output layers
			nn.ConvTranspose2d(filter_sizes[2], filter_sizes[3], 3, stride=2, padding=1, output_padding=1),
			nn.ReLU(inplace=True),
			nn.ConvTranspose2d(filter_sizes[3], filter_sizes[3], 3, stride=2, padding=1, output_padding=1),
			nn.ReLU(inplace=True),
		)

		self.pos_output = nn.Conv2d(filter_sizes[3], 1, kernel_size=1)
		self.cos_output = nn.Conv2d(filter_sizes[3], 1, kernel_size=1)
		self.sin_output = nn.Conv2d(filter_sizes[3], 1, kernel_size=1)
		self.width_output = nn.Conv2d(filter_sizes[3], 1, kernel_size=1)
		self.depth_output = nn.Conv2d(filter_sizes[3], 1, kernel_size=1)

		for m in self.modules():
			if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
				nn.init.xavier_uniform_(m.weight, gain=1)

	def forward(self, x):
		x = self.features(x)
		y = self.grasp(x)
		z = self.depth(x)

		pos_output = self.pos_output(y)
		cos_output = self.cos_output(y)
		sin_output = self.sin_output(y)
		width_output = self.width_output(y)
		depth_output = self.depth_output(z)

		return pos_output, cos_output, sin_output, width_output, depth_output
	
	def predict(self, xc):
		pos_pred, cos_pred, sin_pred, width_pred, depth_pred = self(xc)
		return {
			'pos': pos_pred,
			'cos': cos_pred,
			'sin': sin_pred,
			'width': width_pred,
			'depth': depth_pred
		}

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