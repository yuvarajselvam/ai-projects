import os
import pickle
from matplotlib import pyplot as plt

from util import Counter

for i in range(100):
    os.system(f"python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor,alpha=0.08,runCount={i} "
              f"-x 20 -n 22 -l mediumClassic -q")

totalMap = Counter()
for i in range(100):
    with open(f'Runs/{i}.pickle', 'rb') as f:
        totalMap += Counter(pickle.load(f))

totalMap.divideAll(100)
[print(f'Episode {k}: {v}') for k, v in totalMap.items()]
plt.plot(totalMap.keys(), totalMap.values())
plt.show()

