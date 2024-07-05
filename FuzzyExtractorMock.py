import numpy as np
import random


class FuzzyExtractor:
    def gen(self, bits, locker_size=43, lockers=10000):
        pick_range = range(0, len(bits)-1)
        positions = np.array([random.SystemRandom().sample(
            pick_range, locker_size) for _ in range(lockers)])
        p = []
        for x in range(lockers):
            v_i = np.array([bits[y] for y in positions[x]])
            p.append((v_i, positions[x]))
        return 0, p

    def rep(self, bits, p):
        for v_i, positions in p:
            v_i_prime = np.array([bits[x] for x in positions])
            erroneous_bits = np.count_nonzero(np.bitwise_xor(v_i, v_i_prime))
            if erroneous_bits == 0:
                return True
        return False
