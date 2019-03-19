import numpy as np


def l2(a: np.ndarray, b: np.ndarray) -> float:
	return np.sqrt(np.sum((a - b) ** 2))


def mean_dif_std(arr: np.ndarray) -> float:
	return arr.mean() - arr.std()


def softmax(w, t=1.0) -> float:
	e = np.exp(w / t)
	dist = e / np.sum(e)
	return dist
