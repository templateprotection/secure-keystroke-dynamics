import numpy as np


class FuzzyExtractor:
    def gen(self, bits, locker_size=43, lockers=10000):
        positions = (np.random.random((lockers, locker_size)) * len(bits)).astype(int)
        v_i = bits[positions]
        p = np.stack((v_i, positions), axis=-1)
        return 0, p

    def rep(self, bits, p):
        v_i_all = p[:, :, 0]
        positions_all = p[:, :, 1]
        v_i_prime_all = bits[positions_all]
        erroneous_bits = np.count_nonzero(np.bitwise_xor(v_i_all, v_i_prime_all), axis=1)
        return np.any(erroneous_bits == 0)

