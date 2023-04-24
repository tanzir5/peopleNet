import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(5)
df = pd.DataFrame(np.random.randn(5,5))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(10, 133, as_cmap=True)

with sns.axes_style("white"):
    ax = sns.heatmap(df, cmap=cmap, center=0.00,
                linewidths=.5, )
plt.show()