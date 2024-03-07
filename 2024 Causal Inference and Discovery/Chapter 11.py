import matplotlib.pyplot as plt
import numpy as np
from catenets.models.torch import TARNet, SNet
from econml.metalearners import SLearner, XLearner
from econml.dr import LinearDRLearner
from econml.dml import CausalForestDML
from lightgbm import LGBMRegressor
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error
import pandas as pd
from models.causal_bert_pytorch.CausalBert import CausalBertWrapper

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

#%% CATENets
sample_size = 5000
train_size = 4500
num_features = 20

X = np.random.normal(0, 1, (sample_size, num_features))
T = np.random.binomial(1, 0.5, sample_size)
weights = np.random.gumbel(5, 10, (sample_size, num_features - 1))
y = (50 * T * np.abs(X[:, 0])**1.2) + (weights * X[:, 1:]).sum(axis=1)
y0 = (50 * 0 * np.abs(X[:, 0])**1.2) + (weights * X[:, 1:]).sum(axis=1)
y1 = (50 * 1 * np.abs(X[:, 0])**1.2) + (weights * X[:, 1:]).sum(axis=1)
effect_true = y1[train_size:] - y0[train_size:]

seed = 18
pl.seed_everything(seed)

benchmark_models = {
    'SLearner': SLearner(overall_model=LGBMRegressor()),
    'XLearner': XLearner(models=LGBMRegressor()),
    'DRLearner': LinearDRLearner(),
    'CausalForest': CausalForestDML()
}

benchmark_results = {}
for model_name, model in benchmark_models.items():
    model.fit(X=X[:train_size, :],
              T=T[:train_size],
              Y=y[:train_size])
    effect_pred = model.effect(X[train_size:])
    benchmark_results[model_name] = effect_pred

tarnet = TARNet(n_unit_in=X.shape[1], binary_y=False, n_units_out_prop=32, n_units_r=8, nonlin='selu')
tarnet.fit(X=X[:train_size, :], y=y[:train_size], w=T[:train_size])
effect_pred_tarnet = tarnet.predict(X=X[train_size:, :]).cpu().detach().numpy()
benchmark_results['TARnet'] = effect_pred_tarnet

snet = SNet(n_unit_in=X.shape[1], binary_y=False, n_units_out_prop=32, n_units_r=8, nonlin='selu')
snet.fit(X=X[:train_size, :], y=y[:train_size], w=T[:train_size])
effect_pred_snet = snet.predict(X=X[train_size:, :]).cpu().detach().numpy()
benchmark_results['SNet'] = effect_pred_snet

def plot_results(true_effect, pred_effect, model_name):
    plt.scatter(true_effect, pred_effect, color="#00B0F0")
    plt.plot(np.sort(true_effect), np.sort(true_effect), color='#FF0000', alpha=0.7, label='Perfect model')
    plt.xlabel("True effect")
    plt.ylabel("Predicted effect")
    plt.title(f'{model_name}; MAPE = {mean_absolute_percentage_error(true_effect, pred_effect):.2f}')
    plt.legend()


i = 1
for model_name, results in benchmark_results.items():
    plt.subplot(2, 3, i)
    plot_results(effect_true, results, model_name)

    plt.tight_layout()
    i += 1

#%% causalbert
df = pd.read_csv('./data/manga_processed.csv')
print(df.head())

causal_bert = CausalBertWrapper(batch_size=8, g_weight=0.05, Q_weight=1., mlm_weight=0.05)
causal_bert.train(texts=df.text, confounds=df.has_photo, treatments=df.female_avatar,
                  outcomes=df.upvote, epochs=6)
preds = causal_bert.inference(texts=df.text, confounds=df.has_photo)[0]

ate = np.mean(preds[:, 1] - preds[:, 0])
print(ate)

#%% causality and time series
import pandas as pd
import causalpy as cp
import matplotlib.pyplot as plt

data = pd.read_csv('./data/gt_social_media_data.csv')
print(data.head())

data.index = pd.to_datetime(data.date)
data.drop('date', axis=1, inplace=True)

for i, series in enumerate(data.columns):
    plt.plot(data.index, data[series], label=series.title(), lw=2.5, alpha=0.7)
plt.axvline(pd.to_datetime('2022-10-28'), color='black', lw=3, ls='dotted',
            label='The bird is freed')
plt.xlabel('Day', alpha=0.5)
plt.ylabel('Search volume', alpha=0.5)
plt.legend()

treatment_idx = pd.to_datetime('2022-10-28')
model = cp.pymc_models.WeightedSumFitter()
formula = 'twitter ~ 0 + tiktok + linkedin + instagram'

results = cp.pymc_experiments.SyntheticControl(data, treatment_idx, formula, model=model)

results.plot(plot_predictors=True)
results.summary()


