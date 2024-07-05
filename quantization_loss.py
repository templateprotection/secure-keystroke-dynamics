import random

import numpy as np
import math
import matplotlib.pyplot as plt
import time

from sklearn.metrics import euclidean_distances


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
	best_eer = (best_far + best_frr)/2
	return best_eer*100, best_far*100, best_frr*100, best_thr


def eer_compute(scores_g, scores_i):
	far = []
	frr = []
	ini = min(np.concatenate((scores_g, scores_i)))
	fin = max(np.concatenate((scores_g, scores_i)))
	paso = (fin - ini) / 10000
	threshold = ini - paso
	while threshold < fin + paso:
		far.append(len(np.where(scores_i >= threshold)[0]) / len(scores_i))
		frr.append(len(np.where(scores_g < threshold)[0]) / len(scores_g))
		threshold = threshold + paso

	gap = abs(np.asarray(far) - np.asarray(frr))
	j = np.where(gap == min(gap))[0]
	index = j[0]
	return 100.0 - (((far[index] + frr[index]) / 2) * 100)


def hamming_dist(i1, i2):
	xor_array = i1 ^ i2
	diff = np.count_nonzero(xor_array)
	hd = diff / len(i1)
	return hd


def euclidean_dist(e1, e2):
	return math.sqrt(np.sum((e1 - e2) ** 2))


def normalize_embedding_fw(embedding, all_embeddings):
	feature_mins = np.min(all_embeddings, axis=0)
	feature_maxs = np.max(all_embeddings, axis=0)
	feature_range = feature_maxs - feature_mins
	normalized_embedding = np.clip((embedding - feature_mins) / feature_range, 0, 1)
	return normalized_embedding


def get_w_from_embeddings(embeddings, num_bits_per_feature):
	thresholds = np.linspace(0, 1, num_bits_per_feature + 2)[1:-1]
	bits = np.concatenate(embeddings.reshape(-1, 1) > thresholds)
	return bits


def rand_except(start, stop, exception):
	options = list(range(start, exception)) + list(range(exception + 1, stop))
	return random.choice(options)


def calculate_distances(embeddings, num_bits_per_feature, n1, k, k_test):
	eucs_gen = []
	eucs_imp = []
	hams_gen = []
	hams_imp = []

	n2_users = [n1] + [rand_except(0, len(embeddings), n1) for _ in range(1)]
	for e1 in range(0, len(embeddings[n1])-k, k):
		gallery_embeddings = embeddings[n1][e1:e1 + k]
		emb1 = np.mean(gallery_embeddings, axis=0)
		arr1 = get_w_from_embeddings(emb1, num_bits_per_feature)

		for n2 in n2_users:
			genuine = n1 == n2
			start_idx = e1 + k if genuine else 0
			num_groups = (len(embeddings[n2]) - start_idx) // k_test
			truncate_point = num_groups * k_test + start_idx
			test_embeddings = embeddings[n2][start_idx:truncate_point]
			group_test_embeddings = test_embeddings.reshape((num_groups, k_test, test_embeddings.shape[1]))

			embs2 = np.mean(group_test_embeddings, axis=1)
			arrs2 = [get_w_from_embeddings(emb2, num_bits_per_feature) for emb2 in embs2]

			dists_euc = [np.mean(euclidean_distances(gallery_embeddings, subgroup_test)) for subgroup_test in
						 group_test_embeddings]

			dists_ham = [hamming_dist(arr1, arr2) / num_bits_per_feature for arr2 in arrs2]

			if genuine:
				eucs_gen.extend(dists_euc)
				hams_gen.extend(dists_ham)
			else:
				eucs_imp.extend(dists_euc)
				hams_imp.extend(dists_ham)
	euc_eers.append(calculate_eer(eucs_gen, eucs_imp)[0])
	ham_eers.append(calculate_eer(hams_gen, hams_imp)[0])
	return eucs_gen, eucs_imp, hams_gen, hams_imp


if __name__ == '__main__':
	embeddings = np.load('total_genuine_embs.npy')
	print(embeddings.shape)
	b = 1  # Bits of precision per continuous value
	fusion_gallery = 5  # Number of enrollment templates to store
	fusion_test = 5  # Number of test templates to use

	eucs_gen = []
	eucs_imp = []
	hams_gen = []
	hams_imp = []

	length = len(embeddings)

	euc_eers = []
	ham_eers = []
	t0 = time.time()
	counter = 0
	for n1 in range(length - 1):
		eucs_gen_part, eucs_imp_part, hams_gen_part, hams_imp_part = calculate_distances(embeddings, b, n1, fusion_gallery, fusion_test)
		eucs_gen.extend(eucs_gen_part)
		eucs_imp.extend(eucs_imp_part)
		hams_gen.extend(hams_gen_part)
		hams_imp.extend(hams_imp_part)
		counter += 1
		print(f"{counter}/{length}: EUC={np.mean(euc_eers)} vs HAM={np.mean(ham_eers)}")
	t2 = time.time()
	print(f"Gallery Embeddings: {fusion_gallery}, Test Embeddings: {fusion_test}")

	eucs_gen = np.array(eucs_gen)
	eucs_imp = np.array(eucs_imp)
	hams_gen = np.array(hams_gen)
	hams_imp = np.array(hams_imp)

	euc_eer, euc_far, euc_frr, euc_thr = calculate_eer(eucs_gen, eucs_imp)
	ham_eer, ham_far, ham_frr, ham_thr = calculate_eer(hams_gen, hams_imp)

	print("GLOBAL EUC : far=" + str(euc_far) + "  frr=" + str(euc_frr) + "  thr=" + str(euc_thr))
	print("GLOBAL HAM : far=" + str(ham_far) + "  frr=" + str(ham_frr) + "  thr=" + str(ham_thr))

	print("USER EUC : " + str(np.mean(euc_eers)))
	print("USER HAM : " + str(np.mean(ham_eers)))

	plt.figure(1)
	plt.title("Distribution of scores between continuous embeddings")
	plt.hist(eucs_imp, bins=40, alpha=0.5, color='red', label='Imposter')
	plt.hist(eucs_gen, bins=40, alpha=0.5, color='blue', label='Genuine')
	plt.xlabel('Euclidean Distance')
	plt.ylabel('Density')
	plt.legend()

	plt.figure(2)
	plt.title("Distribution of scores between binary embeddings")
	plt.hist(hams_imp, bins=25, alpha=0.5, color='red', label='Imposter')
	plt.hist(hams_gen, bins=10, alpha=0.5, color='blue', label='Genuine')
	plt.xlabel('Hamming Distance')
	plt.ylabel('Density')
	plt.legend()
	plt.show()
