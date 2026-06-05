import pandas as pd

# 1. Carreguem el fitxer de sentiment que dona problemes
df = pd.read_csv("sentiment_diari_2021_2022.csv")

# 2. Convertim a data i la forgem a un format de text net sense hores (DD/MM/YYYY)
df['Date'] = pd.to_datetime(df['Date'])
df['Date'] = df['Date'].dt.strftime('%d/%m/%Y')

# 3. Guardem sobreescrivint el fitxer
df.to_csv("sentiment_diari_2021_2022.csv", index=False)
print("✅ Fitxer corregit amb format de text net!")