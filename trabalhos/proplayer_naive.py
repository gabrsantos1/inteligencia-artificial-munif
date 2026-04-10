import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
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
    "kills_round", "deaths_round", "kd_ratio",
    "impact", "rating_1"
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

model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("=== Naive Bayes (CS Players) ===")
print(f"Acurácia: {accuracy_score(y_test, y_pred):.2%}")
print()
print(classification_report(y_test, y_pred))