import fuzzywuzzy
import rapidfuzz
from pyctlib import scope
import random
import string

words = [''.join(random.choice(string.ascii_letters + string.digits) for _ in range(10)) for _ in range(10_000)]
samples = words[::len(words) // 100]

with scope("fuzzwuzzy"):
    for sample in samples:
        for word in words:
            fuzzywuzzy.fuzz.partial_ratio(sample, word)

with scope("fuzzwuzzy"):
    for sample in samples:
        for word in words:
            rapidfuzz.fuzz.partial_ratio(sample, word)

with scope("fuzzwuzzy"):
    for sample in samples:
        for word in words:
            fuzzywuzzy.fuzz.ratio(sample, word)

with scope("fuzzwuzzy"):
    for sample in samples:
        for word in words:
            rapidfuzz.fuzz.ratio(sample, word)
