# Copyright (c) 2020-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from abc import ABC, abstractmethod
import numpy as np
import math
from logging import getLogger

logger = getLogger()


class Generator(ABC):
    def __init__(self, params):
        super().__init__()
    
    @abstractmethod
    def generate(self, rng):
        pass

    @abstractmethod
    def evaluate(self, src, tgt, hyp):
        pass


class Sequence(Generator):
    def __init__(self, params, dims):
        super().__init__(params)

        self.maxint = params.maxint
        self.dims = dims
        self.benford = params.benford
        self.train_uniform_lcm = params.train_uniform_lcm
        
       
        self.test_uniform_lcm = params.test_uniform_lcm
        self.max_uniform = params.max_uniform
        self.max_inverse = params.max_inverse

        self.mixture = params.mixture
        self.train_inverse_dist = params.train_inverse_dist
        self.train_sqrt_dist = params.train_sqrt_dist
        self.train_32_dist = params.train_32_dist

        
        self.inverse_dist = np.zeros(self.max_inverse)
        sum = 0.0
        if self.train_sqrt_dist:
            for i in range(self.max_inverse):
                self.inverse_dist[i] = 1 / math.sqrt(i + 1)
                sum += 1 / math.sqrt(i + 1)
        elif self.train_32_dist:
            for i in range(self.max_inverse):
                self.inverse_dist[i] = 1 / (i + 1) * math.sqrt(i + 1)
                sum += 1 / (i + 1) * math.sqrt(i + 1)
        else:
            for i in range(self.max_inverse):
                self.inverse_dist[i] = 1 / (i + 1)
                sum += 1 / (i + 1)
        self.inverse_dist = self.inverse_dist / sum

    def integer_sequence(self, len, rng, type=None, max=None):
        upper = self.maxint if max is None else max
        if type == "train" and self.benford:
            lgs = math.log10(upper) * rng.rand(len)
            return np.int64(10 ** lgs)
        return rng.randint(1, upper + 1, len)

    def generate(self, rng, type=None):
        mix = rng.rand() if (type == "train" and self.mixture > 0.0) else 1.0

        
        if type == "train" and (self.train_inverse_dist or self.train_sqrt_dist):
            L = rng.choice(range(1, self.max_inverse + 1), p=self.inverse_dist)
            # Sample two divisors whose LCM equals L
            divisors = [d for d in range(1, L + 1) if L % d == 0]
            while True:
                d1 = rng.choice(divisors)
                d2 = rng.choice(divisors)
                if abs(d1 * d2) // math.gcd(d1, d2) == L:
                    inp = [d1, d2]
                    break
            out = L

        # Uniform-LCM distribution case
        elif (type == "train" and (self.train_uniform_lcm or mix < self.mixture)) or type == "test":
            L = rng.randint(1, self.max_uniform + 1)
            divisors = [d for d in range(1, L + 1) if L % d == 0]
            while True:
                d1 = rng.choice(divisors)
                d2 = rng.choice(divisors)
                if abs(d1 * d2) // math.gcd(d1, d2) == L:
                    inp = [d1, d2]
                    break
            out = L

        # Default: random pair, compute their LCM
        else:
            inp = self.integer_sequence(2, rng, type)
            # Compute LCM via gcd fallback
            out = abs(int(inp[0]) * int(inp[1])) // math.gcd(int(inp[0]), int(inp[1]))

        return inp, out

    def evaluate(self, src, tgt, hyp):
        return 0, 0, 0, 0
