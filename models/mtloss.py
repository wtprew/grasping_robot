import torch
import torch.nn as nn

class MultiTaskLoss(nn.Module):                                                                                                                                              
	def __init__(self, num_tasks):                                                                                                                                           
		super(MultiTaskLoss, self).__init__()                                                                                                                                
		self.log_var = torch.nn.Parameter(data=torch.zeros((num_tasks,)), requires_grad=True)                                                                                
	
	def get_var(self):                                                                                                                                                       
		return torch.exp(-self.log_var)                                                                                                                                      
	
	def forward(self, *losses):                                                                                                                                              
		losses = torch.stack(losses)                                                                                                                                         
		var = self.get_var()                                                                                                                                                 
		return torch.sum(var * losses + self.log_var), self.log_var.data.tolist()