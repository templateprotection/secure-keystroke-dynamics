import time
import numpy as np

from FuzzyExtractorMock import FuzzyExtractor


def normalize_embedding_fw(embedding, all_embeddings):
	feature_mins = np.min(all_embeddings, axis=0)
	feature_maxs = np.max(all_embeddings, axis=0)
	feature_range = feature_maxs - feature_mins
	normalized_embedding = np.clip((embedding-feature_mins) / feature_range, 0, 1)
	return normalized_embedding


def get_w_from_embeddings_fw(embedding, all_embs, num_bits_per_feature):
	normalized_embedding = normalize_embedding_fw(embedding, all_embs)
	w = []
	for val in normalized_embedding:
		thresholds = np.linspace(0, 1, num_bits_per_feature + 2)[1:-1]
		bits = val > thresholds
		w.extend(bits)
	return np.array(w)


if __name__ == "__main__":
	all_embs = np.load("total_genuine_embs.npy")
	embeddings = dict()
	for user_num in range(len(all_embs)):
		user_id = str(user_num)
		embeddings[user_id] = []
		for emb in all_embs[user_num]:
			embeddings[user_id].append(emb)
	
	num_bits_per_feature = 1

	"""
	NEW RESULTS TO INPUT with K_TEST
	
	27/100/1/1: FAR=0.107, FRR=0.16583416583416583   eer=13.5  [TIME=0.004, 0.005, 0.005]
	34/100/3/3: FAR=0.07592675301473872, FRR=0.07092907092907093   eer = 7.34 [TIME=0.004, 0.005, 0.006]
	39/100/5/5: FAR=0.051, FRR=0.04395604395604396    eer = 4.75 [TIME=0.004, 0.005, 0.006]
	
	38/1,000/1/1 : FAR=0.11, FRR=0.13586413586413587   eer = 12.39 [TIME=0.007, 0.016, 0.06]
	50/1,000/3/3: FAR=0.05, FRR=0.055944055944055944   eer = 5.29 [TIME=0.006, 0.019, 0.08]
	58/1,000/5/5: FAR=0.036, FRR=0.03996003996003996   eer = 3.79  [TIME=0.006, 0.022, 0.09]

	50/10,000/1/1: FAR=0.097, FRR=0.11092837498984987, eer = 10.35 [TIME=0.039, 0.153, 0.80]
	66/10,000/3/3: FAR=0.048, FRR=0.053946053946053944 eer = 5.09 [TIME=0.024, 0.19, 1.01]
	79/10,000/5/5: FAR=0.032, FRR=0.03196803196803197   eer = 3.19 [TIME=0.025, 0.23, 1.21]
	
	
	61/100,000/1/1: FAR=0.1075, FRR=0.009951029390192309    eer = 10.35  [TIME=0.37, 1.78, 9.17]
	82/100,000/3/3: FAR=0.048, FRR=0.04295704295704296     eer = 4.55  [TIME=0.18, 2.38, 12.32]
	96/100,000/5/5: FAR=0.035, FRR=0.029970029970029972     eer = 3.25  [TIME=0.15, 2.72, 14.48]

	"""

	locker_size = 34  # Number of bits per locker 'k'
	lockers = 10000  # Number of lockers 'n'
	g = 3  # Number of enrollment embeddings to fuse
	g_test = 3  # Number of test embeddings to fuse
	print("===Params===")
	print(locker_size)
	print(lockers)
	print(g)
	print(g_test)
	print("============")
	
	# Matching Results
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
		for e1 in range(0, g, g):  # Only test one embedding per user
			emb1 = np.mean(embeddings[name1][e1:e1 + g], axis=0)
			arr1 = get_w_from_embeddings_fw(emb1, all_embs, num_bits_per_feature)
			t0 = time.time()
			_, p = fe.gen(arr1, locker_size=locker_size, lockers=lockers)
			t1 = time.time()
			enr_times.append(t1-t0)
			for n2 in range(n1, min(n1+2, length)):
				name2 = names[n2]
				genuine = name1 == name2
				s_ind = e1 + g if genuine else 0
				for e2 in range(s_ind, s_ind + g_test, g_test):
					print(name1 + "_" + str(e1) + " = " + name2 + "_" + str(e2) + "? ",end="")
					t0 = time.time()
					emb2 = np.mean(embeddings[name2][e2:e2 + g_test], axis=0)
					arr2 = get_w_from_embeddings_fw(emb2, all_embs, num_bits_per_feature)
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
