from abc import ABC, abstractmethod

import numpy as np
import random


class EmbeddingNormalizer(ABC):
    @abstractmethod
    def get_binary_embedding(self, embedding):
        pass

    def __str__(self):
        return self.__class__.__name__


class MinMaxEmbeddingNormalizer(EmbeddingNormalizer):
    def __init__(self, all_embeddings, b=1):
        self.__min_values = np.min(all_embeddings, axis=0).min(axis=0)
        self.__max_values = np.max(all_embeddings, axis=0).max(axis=0)
        self.__range_values = self.__max_values - self.__min_values
        self.__binary_thresholds = np.linspace(0, 1, b + 2)[1:-1]

    def normalize(self, embedding):
        return np.clip((embedding - self.__min_values) / self.__range_values, 0, 1)

    def get_binary_embedding(self, embedding):
        w = []
        for val in self.normalize(embedding):
            bits = val > self.__binary_thresholds
            w.extend(bits)
        return np.array(w)


class QuantileEmbeddingNormalizer(EmbeddingNormalizer):
    def __init__(self, all_embeddings, quantile=0.75):
        self.__quantile = quantile
        self.__quantile_thresholds = np.quantile(all_embeddings, quantile, axis=(0, 1))

    def get_binary_embedding(self, embedding):
        w = embedding > self.__quantile_thresholds
        return w


class MeanEmbeddingNormalizer(QuantileEmbeddingNormalizer):
    def __init__(self, all_embeddings):
        super().__init__(all_embeddings, 0.5)


def calculate_eer(gen_scores, imp_scores, num_samples=1000):
    if len(gen_scores) + len(imp_scores) < num_samples:
        thresholds = np.array(list(set(gen_scores) | set(imp_scores)))
    else:
        thresholds = np.linspace(min(gen_scores), max(imp_scores), num_samples)

    gen_scores = np.array(gen_scores)
    imp_scores = np.array(imp_scores)

    fars = np.array([len(imp_scores[imp_scores <= t]) / len(imp_scores) for t in thresholds])
    frrs = np.array([len(gen_scores[gen_scores > t]) / len(gen_scores) for t in thresholds])

    best_idx = np.argmin(np.abs(fars - frrs))
    best_thr = thresholds[best_idx]
    best_far = fars[best_idx]
    best_frr = frrs[best_idx]
    best_eer = (best_far + best_frr) / 2
    return best_eer * 100, best_far * 100, best_frr * 100, best_thr


def hamming_dist(i1, i2):
    xor_array = i1 ^ i2
    diff = np.count_nonzero(xor_array)
    hd = diff / len(i1)
    return hd


def rand_except(start, stop, exception):
    options = list(range(start, exception)) + list(range(exception + 1, stop))
    return random.choice(options)
