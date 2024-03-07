import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

noise_level = 0.2
num_samples = 1000


def plot_vars(title, a, b, c):
    xvars = [a, a, b]
    yvars = [b, c, c]
    xlabs = ['a', 'a', 'b']
    ylabs = ['b', 'c', 'c']

    num_pairs = min(len(xvars), len(yvars))

    fig, axs = plt.subplots(2, 2)
    for idx, ax in enumerate(axs.flat):
        if idx >= num_pairs:
            ax.remove()
            continue
        ax.scatter(xvars[idx], yvars[idx], alpha=0.2)
        ax.set_xlabel(f'{xlabs[idx]}')
        ax.set_ylabel(f'{ylabs[idx]}')
    plt.suptitle(title)
    fig.tight_layout()


def gen_linear_model(a, b):
    X = pd.DataFrame(np.vstack([a, b]).T, columns=['A', 'B'])
    X = sm.add_constant(X, prepend=True)
    model = sm.OLS(c, X)
    results = model.fit()
    print(results.summary())

# the chain dataset
a = np.random.randn(num_samples)
b = a + noise_level * np.random.randn(num_samples)
c = b + noise_level * np.random.randn(num_samples)

plot_vars('Chain - scatterplots', a, b, c)
gen_linear_model(a, b)

# generating the fork dataset
b = np.random.randn(num_samples)
a = b + noise_level * np.random.randn(num_samples)
c = b + noise_level * np.random.randn(num_samples)

plot_vars('Fork - scatterplots', a, b, c)
gen_linear_model(a, b)

# generating the collider dataset
a = np.random.randn(num_samples)
c = np.random.randn(num_samples)
b = a + c + noise_level * np.random.randn(num_samples)

plot_vars('Collider - scatterplots', a, b, c)
gen_linear_model(a, b)
