import random

import numpy as np
import math
import matplotlib.pyplot as plt
import time

from sklearn.metrics import euclidean_distances

from QuantizationUtils import rand_except, hamming_dist, calculate_eer, MinMaxEmbeddingNormalizer


def calculate_distances(embeddings, num_bits_per_feature, n1, k, k_test, embedding_normalizer):
	global all_embs
	eucs_gen = []
	eucs_imp = []
	hams_gen = []
	hams_imp = []

	n2_users = [n1] + [rand_except(0, len(embeddings), n1) for _ in range(1)]
	for e1 in range(0, len(embeddings[n1])-k, k):
		gallery_embeddings = embeddings[n1][e1:e1 + k]
		emb1 = np.mean(gallery_embeddings, axis=0)
		arr1 = embedding_normalizer.get_binary_embedding(emb1)

		for n2 in n2_users:
			genuine = n1 == n2
			start_idx = e1 + k if genuine else 0
			num_groups = (len(embeddings[n2]) - start_idx) // k_test
			truncate_point = num_groups * k_test + start_idx
			test_embeddings = embeddings[n2][start_idx:truncate_point]
			group_test_embeddings = test_embeddings.reshape((num_groups, k_test, test_embeddings.shape[1]))

			embs2 = np.mean(group_test_embeddings, axis=1)
			arrs2 = [embedding_normalizer.get_binary_embedding(emb2) for emb2 in embs2]

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
	embeddings = np.load('embeddings_1k.npy')
	print(embeddings.shape)
	b = 1  # Bits of precision per continuous value
	fusion_gallery = 5  # Number of enrollment templates to store
	fusion_test = 5  # Number of test templates to use

	embedding_normalizer = MinMaxEmbeddingNormalizer(embeddings, b)

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
		eucs_gen_part, eucs_imp_part, hams_gen_part, hams_imp_part = calculate_distances(embeddings, b, n1, fusion_gallery, fusion_test, embedding_normalizer)
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
