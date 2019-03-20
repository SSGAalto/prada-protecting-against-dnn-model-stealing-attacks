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

from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
from flask import Flask, request
from scipy import misc

import growing_set as gs
import growing_set_ops as gso
import model
import model_ops as mops


def serve_model(delta: float, oracle_path: str, model_class: model):
	gd_agent = gs.GrowingDistanceAgent(
		delta=delta,
		dist_metric=gso.l2,
		thr_update_rule=gso.mean_dif_std)

	allowed_extensions = ["jpg", "png", "ppm"]
	app = Flask(__name__)

	oracle = mops.load_server(oracle_path, model_class=model_class)
	oracle_predict = mops.model_handle(oracle)

	@app.route("/predict", methods=["POST"])
	def upload_image():
		if request.method == "POST":
			img_file = request.files['payload']
			if img_file and img_file.filename[-3:] in allowed_extensions:
				img_query = to_matrix(img_file)
				logits = oracle_predict(img_query)
				target_class = np.argmax(gso.softmax(logits))
				attacker_present = gd_agent.single_query(img_query, target_class)
				res = shuffle_max_logits(logits, 3) if attacker_present else logits
				return str(res)

	app.run(port=8080, host="localhost")


def to_matrix(img_file) -> np.ndarray:
	return misc.imread(img_file)


def shuffle_max_logits(logits: np.ndarray, n: int) -> np.ndarray:
	# simple defense mechanism that shuffles top n logits
	logits = logits.squeeze()
	idx = logits.argsort()[-n:][::-1]
	max_elems = logits[idx]
	np.random.shuffle(max_elems)
	for i, e in zip(idx, max_elems):
		logits[i] = e
	return logits
