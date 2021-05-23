import numpy as np
import pandas as pd
from src.CoFlux.predict_realization import *
import matplotlib.pyplot as plt
import time

t1 = time.time()
x = 0
listA = list()
listB = list()
listC = list()
for i in range(800):
    listA.append(np.sin(x * np.pi))
    listB.append(np.sin(x * np.pi - np.pi / 2))
    x += 0.1
plt.plot(np.arange(0, 800), np.array(listA), label="A")
plt.plot(np.arange(0, 800), np.array(listB), label="B")
plt.legend()
plt.show()