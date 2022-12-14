import torch
import torch.nn as nn
import torch.nn.functional as F


class GraspModel(nn.Module):
	"""
	An abstract model for grasp network in a common format.
	"""

	def __init__(self):
		super(GraspModel, self).__init__()

	def forward(self, x_in):
		raise NotImplementedError()

	def compute_loss(self, xc, yc, newloss=False):
		y_pos, y_cos, y_sin, y_width = yc
		pos_pred, cos_pred, sin_pred, width_pred = self(xc)

		if newloss == False:
			p_loss = F.smooth_l1_loss(pos_pred, y_pos)
			cos_loss = F.smooth_l1_loss(cos_pred, y_cos)
			sin_loss = F.smooth_l1_loss(sin_pred, y_sin)
			width_loss = F.smooth_l1_loss(width_pred, y_width)
		else:
			p_loss = F.mse_loss(pos_pred, y_pos)
			cos_loss = F.mse_loss(torch.mul(y_pos, cos_pred), torch.mul(y_pos, y_cos))
			sin_loss = F.mse_loss(torch.mul(y_pos, sin_pred), torch.mul(y_pos, y_sin))
			width_loss = F.mse_loss(torch.mul(y_pos, width_pred), torch.mul(y_pos, y_width))

		return {
			'loss': p_loss + cos_loss + sin_loss + width_loss,
			'losses': {
				'p_loss': p_loss,
				'cos_loss': cos_loss,
				'sin_loss': sin_loss,
				'width_loss': width_loss
			},
			'pred': {
				'pos': pos_pred,
				'cos': cos_pred,
				'sin': sin_pred,
				'width': width_pred
			}
		}

	def predict(self, xc):
		pos_pred, cos_pred, sin_pred, width_pred = self(xc)
		return {
			'pos': pos_pred,
			'cos': cos_pred,
			'sin': sin_pred,
			'width': width_pred
		}


class ResidualBlock(nn.Module):
	"""
	A residual block with dropout option
	"""

	def __init__(self, in_channels, out_channels, kernel_size=3, dropout=False, prob=0.0):
		super(ResidualBlock, self).__init__()
		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
		self.bn1 = nn.BatchNorm2d(in_channels)
		self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
		self.bn2 = nn.BatchNorm2d(in_channels)

		self.dropout = dropout
		self.dropout1 = nn.Dropout(p=prob)

	def forward(self, x_in):
		x = self.bn1(self.conv1(x_in))
		x = F.relu(x)
		# if self.dropout:
			# x = self.dropout1(x)
		x = self.bn2(self.conv2(x))
		return x + x_in
