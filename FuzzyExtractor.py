import hmac
import numpy as np
from hashlib import sha512


def generate_sample(length=0, size=32):
    return np.random.randint(0, 256, (length, size), dtype=np.uint8)


def check_result(res):
    pad_len = int(len(res) - len(res) / 2)
    return all(v == 0 for v in res[:pad_len])


def xor(b1, b2):
    return bytearray([x ^ y for x, y in zip(b1, b2)])


class FuzzyExtractor:
    def __init__(self, _hash=sha512):
        self.hash = _hash

    def gen(self, bits, locker_size=43, lockers=1000000):
        length = self.hash().digest_size
        rand_len = length // 2
        pad_len = length - rand_len
        r = generate_sample(size=rand_len).tobytes()
        zeros = bytes(pad_len)
        r_padded = zeros + r

        seeds = generate_sample(length=lockers, size=16)
        positions = np.random.randint(0, len(bits), (lockers, locker_size))
        v_i = np.take(bits, positions)

        p = []
        for x in range(lockers):
            seed = seeds[x].tobytes()
            v_i_bytes = v_i[x].tobytes()
            h = hmac.new(seed, v_i_bytes, self.hash).digest()
            c_i = xor(r_padded, h)
            p.append((c_i, positions[x], seed))

        return r, p

    def rep(self, bits, p):
        for c_i, positions, seed in p:
            v_i = np.take(bits, positions).tobytes()
            h = hmac.new(seed, v_i, self.hash).digest()
            res = xor(c_i, h)
            if check_result(res):
                key_len = len(res) // 2
                return res[key_len:]

        return None
