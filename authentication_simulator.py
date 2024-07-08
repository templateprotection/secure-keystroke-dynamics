import time
import numpy as np

from FuzzyExtractorMock import FuzzyExtractor
from QuantizationUtils import MinMaxEmbeddingNormalizer, MeanEmbeddingNormalizer, QuantileEmbeddingNormalizer

if __name__ == "__main__":
	all_embs = np.load("embeddings_1k.npy")
	print(all_embs.shape)
	embeddings = dict()
	for user_num in range(len(all_embs)):
		user_id = str(user_num)
		embeddings[user_id] = []
		for emb in all_embs[user_num]:
			embeddings[user_id].append(emb)


	"""
		29/100/1/1: FAR=0.129, FRR=0.13786213786213786, eer=13.34
		40/100/3/3: FAR=0.077, FRR=0.07592407592407592, eer=7.64
		45/100/5/5: FAR=0.047, FRR=0.04595404595404595, eer=4.64

		42/1,000/1/1: FAR=0.128, FRR=0.11688311688311688, eer=12.24
		58/1,000/3/3: FAR=0.066, FRR=0.06893106893106893, eer=6.74
		66/1,000/5/5: FAR=0.043, FRR=0.04095904095904096, eer=4.19

		56/10,000/1/1: FAR=0.117, FRR=0.11788211788211789, eer=11.74
		77/10,000/3/3: FAR=0.056, FRR=0.05194805194805195, eer=5.39 
		86/10,000/5/5: FAR=0.037, FRR=0.028971028971028972, eer=3.29 


		70/100,000/1/1: FAR=0.112, FRR=0.11388611388611389, eer=11.29
		95/100,000/3/3: FAR=0.052, FRR=0.04695304695304695, eer=4.94 
		109/100,000/5/5: FAR=0.033, FRR=0.03796203796203796, eer=2.54 
	"""

	locker_size = 86  # Number of bits per locker 'k'
	lockers = 10000  # Number of lockers 'n'
	gallery_fusion = 5  # Number of enrollment embeddings to fuse
	test_fusion = 5  # Number of test embeddings to fuse
	b = 1  # Number of precision bits per feature

	# MinMaxEmbeddingNormalizer: MinMax scale all embeddings between 0 and 1, and binarize based on 0.5.
	# QuantileEmbeddingNormalizer: Binarize based on a specified quantile of all embeddings.
	# MeanEmbeddingNormalizer: QuantileEmbeddingNormalizer with quantile=0.5
	embedding_normalizer = MinMaxEmbeddingNormalizer(all_embs)

	print("===Params===")
	print(locker_size)
	print(lockers)
	print(gallery_fusion)
	print(test_fusion)
	print(embedding_normalizer)
	print("============")

	all_recovered = []
	all_genuines = []
	
	frs = 0
	fas = 0
	imps = 1
	gens = 1
	
	fe = FuzzyExtractor()
	names = list(embeddings.keys())
	length = len(names)

	enr_times = []
	gen_auth_times = []
	imp_auth_times = []
	for n1 in range(0, len(embeddings)):
		name1 = names[n1]
		for e1 in range(0, gallery_fusion, gallery_fusion):  # Only test one embedding per user
			emb1 = np.mean(embeddings[name1][e1:e1 + gallery_fusion], axis=0)
			arr1 = embedding_normalizer.get_binary_embedding(emb1)
			t0 = time.time()
			_, p = fe.gen(arr1, locker_size=locker_size, lockers=lockers)
			t1 = time.time()
			enr_times.append(t1-t0)
			for n2 in range(n1, min(n1+2, length)):
				name2 = names[n2]
				genuine = name1 == name2
				s_ind = e1 + gallery_fusion if genuine else 0
				for e2 in range(s_ind, s_ind + test_fusion, test_fusion):
					print(name1 + "_" + str(e1) + " = " + name2 + "_" + str(e2) + "? ", end="")
					t0 = time.time()
					emb2 = np.mean(embeddings[name2][e2:e2 + test_fusion], axis=0)
					arr2 = embedding_normalizer.get_binary_embedding(emb2)
					match = fe.rep(arr2, p)
					t1 = time.time()
					if genuine:
						gen_auth_times.append(t1 - t0)
						gens += 1
						if not match:
							frs += 1
					else:
						imp_auth_times.append(t1 - t0)
						imps += 1
						if match:
							fas += 1

					far = fas / imps
					frr = frs / gens
					match_str = "Succeeded" if match else "Failed"
					match_str += " FAR=" + str(far) + ", FRR=" + str(frr) + " TIME_G=" + str(np.mean(gen_auth_times)) + " TIME_I=" + str(np.mean(imp_auth_times)) + " TIME_E=" + str(np.mean(enr_times))
					print(match_str)
