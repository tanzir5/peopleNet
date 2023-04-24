import numpy as np
import pandas as pd
from autoimpute.imputations import SingleImputer, MultipleImputer, MiceImputer

x = np.random.rand(40)
y = np.random.rand(40)
z = ['good'] * 30 + ['bad'] * 5 + ['sag'] * 3 + ['lol'] * 2
z = [1] * 30 + [2] * 5 + [3] * 3 + [4] * 2
print(len(z))
y[:30] = 0
z[39] = np.nan
y[2] = np.nan
z[0] = np.nan
z[5] = np.nan
z[6] = np.nan
x[2] = np.nan
data_miss = pd.DataFrame({"majhari": z, "valo": y, "kharap": x})

mi = MiceImputer(n = 5, return_list=True, 
                      strategy={'majhari':'multinomial logistic', 'valo':'default predictive'})
mi_data_full = mi.fit_transform(data_miss)

print(type(data_miss), type(mi_data_full[4][1]))
print(mi_data_full[4][1])
print("*"*100)
print(data_miss)
