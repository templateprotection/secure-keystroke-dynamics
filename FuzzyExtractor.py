import hmac
import numpy as np
import random
from hashlib import sha512


def generate_sample(length=0, size=32):
    rand_gen = random.SystemRandom()
    if length == 0:
        return bytearray([rand_gen.randint(0, 255) for _ in range(int(size))])
    else:
        samples = []
        for x in range(length):
            samples.append(
                bytearray([rand_gen.randint(0, 255) for _ in range(int(size))]))
        return samples


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

        rand_len = int(length / 2)
        pad_len = length - rand_len
        r = generate_sample(size=rand_len)
        zeros = bytearray([0 for _ in range(pad_len)])
        r_padded = zeros + r

        seeds = generate_sample(length=lockers, size=16)
        pick_range = range(0, len(bits) - 1)
        positions = np.array([random.SystemRandom().sample(
            pick_range, locker_size) for _ in range(lockers)])
        p = []
        for x in range(lockers):
            v_i = np.array([bits[y] for y in positions[x]])
            seed = seeds[x]
            h = bytearray(hmac.new(seed, v_i, self.hash).digest())
            c_i = xor(r_padded, h)
            p.append((c_i, positions[x], seed))
        return r, p

    def rep(self, bits, p):
        count = 0
        for c_i, positions, seed in p:
            v_i = np.array([bits[x] for x in positions])
            h = bytearray(hmac.new(seed, v_i, self.hash).digest())

            # in gen, we say c_i = res xor h
            res = xor(c_i, h)
            if check_result(res):
                key_len = int(len(res) / 2)
                return res[key_len:]
            count += 1
        return None
