# %%
import pandas as pd

# %%

df = pd.read_csv("stackline_sales.csv")

df.head()
# %%

print(", ".join(df.columns))
# %%