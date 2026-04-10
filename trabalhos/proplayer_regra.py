import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

url = "https://docs.google.com/spreadsheets/d/1AVaJevzo0aPWNZ6o4GooZ-a4UAp1aRbKHDlFSf6zit0/export?format=csv"
df = pd.read_csv(url)

df = df.drop(columns=["Unnamed: 0"], errors="ignore")

for col in ["hs_percentage", "kast"]:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace('%', '', regex=False)
        .replace('-', None)
    )
    df[col] = pd.to_numeric(df[col], errors="coerce") / 100

colunas_numericas = [
    "kills_round", "deaths_round", "kd_ratio",
    "impact", "rating_1"
]

for col in colunas_numericas:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna()

def categorizar(df):
    df_cat = pd.DataFrame()

    df_cat["kills"] = df["kills_round"].apply(lambda x: "kills_alto" if x >= 0.7 else "kills_baixo")
    df_cat["deaths"] = df["deaths_round"].apply(lambda x: "deaths_alto" if x >= 0.75 else "deaths_baixo")
    df_cat["kd"] = df["kd_ratio"].apply(lambda x: "kd_alto" if x >= 1.1 else "kd_baixo")
    df_cat["impact"] = df["impact"].apply(lambda x: "impact_alto" if x >= 1 else "impact_baixo")
    df_cat["hs"] = df["hs_percentage"].apply(lambda x: "hs_alto" if x >= 0.5 else "hs_baixo")

    df_cat["performance"] = df["rating_1"].apply(
        lambda x: "elite" if x >= 1.1 else "normal"
    )

    return df_cat

df_cat = categorizar(df)

transactions = []
for _, row in df_cat.iterrows():
    transactions.append(list(row.values))

te = TransactionEncoder()
te_array = te.fit_transform(transactions)
df_encoded = pd.DataFrame(te_array, columns=te.columns_)

frequent_itemsets = apriori(df_encoded, min_support=0.1, use_colnames=True)
frequent_itemsets = frequent_itemsets.sort_values("support", ascending=False)

print("=== Itemsets Frequentes ===")
print(frequent_itemsets.head(10).to_string(index=False))

rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)
rules = rules.sort_values("lift", ascending=False)

print("\n=== Regras de Associação ===")
print(
    rules[["antecedents", "consequents", "support", "confidence", "lift"]]
    .head(10)
    .to_string(index=False)
)