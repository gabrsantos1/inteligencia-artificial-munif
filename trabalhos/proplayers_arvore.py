import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, classification_report

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
    "assists_round", "deaths_round", "dmg_round", "gnd_dmg_round",
    "impact", "kd_ratio", "kills_round",
    "rating_1", "rounds_played"
]

for col in colunas_numericas:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna()

df["High_Performance"] = (df["rating_1"] >= 1.1).astype(int)

X = df[
    [
        "kills_round",
        "deaths_round",
        "kd_ratio",
        "hs_percentage",
        "impact",
        "kast",
    ]
]

y = df["High_Performance"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("=== Árvore de Decisão (CS Players) ===")
print(f"Acurácia: {accuracy_score(y_test, y_pred):.2%}")
print()
print(classification_report(y_test, y_pred))

# resultados = X_test.copy()
# resultados["Real"] = y_test.values
# resultados["Previsto"] = y_pred

# print("\n=== Previsões ===")
# print(resultados.head(10))

# print("\n=== Regras da Árvore ===")
# regras = export_text(model, feature_names=list(X.columns))
# print(regras)
