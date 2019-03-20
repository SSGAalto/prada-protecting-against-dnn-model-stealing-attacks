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

import numpy as np


def l2(a: np.ndarray, b: np.ndarray) -> float:
	return np.sqrt(np.sum((a - b) ** 2))


def mean_dif_std(arr: np.ndarray) -> float:
	return arr.mean() - arr.std()


def softmax(w, t=1.0) -> float:
	e = np.exp(w / t)
	dist = e / np.sum(e)
	return dist
