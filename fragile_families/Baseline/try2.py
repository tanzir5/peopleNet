import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

df = pd.DataFrame({'A': [2, 3, np.nan, 5],
                   'B': [np.nan, 9, 16, 25],
                   'C': ['good', np.nan, 'bad', 'good']})

# Setting the random_state argument for reproducibility
imputer = IterativeImputer(random_state=42)
imputed = imputer.fit_transform(df)
df_imputed = pd.DataFrame(imputed, columns=df.columns)
print(df_imputed)