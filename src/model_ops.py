# Author: Samuel Marchal samuel.marchal@aalto.fi Sebastian Szyller sebastian.szyller@aalto.fi Mika Juuti mika.juuti@aalto.fi
# Copyright 2019 Secure Systems Group, Aalto University, https://ssg.aalto.fi
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from torch import autograd
from torch import nn
import numpy as np
import torch


def load_server(load_location='', model_class=None):
	net = model_class()
	net.load_state_dict(torch.load(load_location))
	return net


# function responsible for making the prediction with your model
# adjust to your needs; by default, works with usual MNIST setup
def model_handle(oracle: nn.Module) -> callable:
	def predict(img_query: np.ndarray) -> np.ndarray:
		img_torch = torch.from_numpy(img_query).view(1, 1, 28, 28).float()
		img_torch = to_range(img_torch, img_torch.min(), img_torch.max(), -1., 1.)
		return oracle(autograd.Variable(img_torch)).data.cpu().numpy()
	return predict


# adjust to the normalization range
def to_range(x, old_min, old_max, new_min, new_max):
	old_range = (old_max - old_min)
	new_range = (new_max - new_min)
	return (((x - old_min) * new_range) / old_range) + new_min
