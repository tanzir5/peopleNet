import numpy as np
import pandas as pd
from scipy.stats import norm, binom
from autoimpute.imputations import SingleImputer, MultipleImputer
from autoimpute.visuals import plot_imp_dists, plot_imp_boxplots, plot_imp_swarm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
sns.set(context="talk", rc={'figure.figsize':(11.7,8.27)})

# helper functions used throughout this project
print_header = lambda msg: print(f"{msg}\n{'-'*len(msg)}")

# seed to follow along
np.random.seed(654654)

# generate 400 data points
N = np.arange(400)

# helper function for this data
vary = lambda v: np.random.choice(np.arange(v))

# create correlated, random variables
a = 2
b = 1/2
eps = np.array([norm(0, vary(30)).rvs() for n in N])
y = (a + b*N + eps) / 100
x = (N + norm(10, vary(250)).rvs(len(N))) / 100

# 30% missing in y
y[binom(1, 0.3).rvs(len(N)) == 1] = np.nan

# collect results in a dataframe
data_miss = pd.DataFrame({"y": y, "x": x})
sns.scatterplot(x="x", y="y", data=data_miss)
#plt.show()

si = SingleImputer()
si_data_full = si.fit_transform(data_miss)

# print the results
print_header("Results from SingleImputer running PMM on column y one time")
conc = pd.concat([data_miss.head(20), si_data_full.head(20)], axis=1)
conc.columns = ["x", "y_orig", "x_imp", "y_imp"]
conc[["x", "y_orig", "y_imp"]]