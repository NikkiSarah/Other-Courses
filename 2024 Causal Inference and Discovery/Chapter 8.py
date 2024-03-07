import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl_backend = mpl.get_backend()
if mpl_backend != "Qt5Agg":
    mpl.use("Qt5Agg")
else:
    pass

sample_size = 1000

T = np.random.uniform(20, 100, sample_size)
Y = T + np.random.uniform(0, 40, sample_size)
C = (T + Y < 100).astype('int')

df = pd.DataFrame(np.vstack([T, Y, C]).T, columns=['T', 'Y', 'C'])

plt.figure()
plt.hist(df[df.C == 1]['Y'], label="Returned", color='#00B0F0', alpha=0.5)
plt.hist(df[df.C == 0]['Y'], label="Didn't return", color='#FF0000', alpha=0.5, bins=25)
plt.xlabel('$Damage$ $severity$', alpha=0.5, fontsize=12)
plt.ylabel('$Frequency$', alpha=0.5, fontsize=12)
plt.legend()
