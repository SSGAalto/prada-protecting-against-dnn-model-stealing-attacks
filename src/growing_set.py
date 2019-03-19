from typing import Dict
from scipy import stats
import numpy as np


class GrowingDistanceAgent:
    def __init__(self, shapiro_threshold: float, dist_metric: callable,
                 thr_update_rule: callable):

        self.shapiro_threshold = shapiro_threshold
        self.dist_metric = dist_metric
        self.threshold_update_rule = thr_update_rule

        self.growing_set = {}
        self.growing_set_dists = {}
        self.set_size_ot = []
        self.gs_size = 0
        self.queries_processed = 0
        self.threshold = {}
        self.attacker_present = 0
        self.attacker_present_ot = []

        self.distri_stats = {}
        self.distri_stats["shapiro"] = []
        self.input_distances = {}

    def __update_threshold(self, gs_distances: np.ndarray, current_threshold: float) -> float:
        threshold_candidate = self.threshold_update_rule(gs_distances)
        return max(threshold_candidate, current_threshold)

    def __reject_outliers(self, data, m=2):
        return data[abs(data - np.mean(data)) < m * np.std(data)]

    def single_query(self, img_query: np.ndarray, target_class: np.ndarray) -> int:

        if target_class not in self.growing_set:
            self.growing_set[target_class] = [img_query]
            self.threshold[target_class] = 0
            self.growing_set_dists[target_class] = [0]
            self.gs_size += 1
        else:
            dists_ = np.asarray([self.dist_metric(x, img_query) for x in self.growing_set[target_class]])
            min_ = np.min(dists_)

            if target_class not in self.input_distances:
                self.input_distances[target_class] = [min_]
            else:
                self.input_distances[target_class].append(min_)

            if min_ > self.threshold[target_class]:
                self.growing_set[target_class].append(img_query)
                self.gs_size += 1
                self.growing_set_dists[target_class].append(min_)
                self.threshold[target_class] = \
                    self.__update_threshold(np.asarray(self.growing_set_dists[target_class]),
                                            self.threshold[target_class])

        self.set_size_ot.append(self.gs_size)

        if self.queries_processed % 10 == 0:
            dists = []
            for classe_, dist in self.input_distances.items():
                if len(dist) >= 10:
                    dists.extend(dist)

            dists = self.__reject_outliers(np.asarray(dists), 3)
            if len(dists) > 100:
                k2_2, p_2 = stats.shapiro(dists)
                self.distri_stats["shapiro"].append(k2_2)
                self.attacker_present = 1 if k2_2 < self.shapiro_threshold else 0

        self.attacker_present_ot.append(self.attacker_present)
        self.queries_processed += 1
        return self.attacker_present
