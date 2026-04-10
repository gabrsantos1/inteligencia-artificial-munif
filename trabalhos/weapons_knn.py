import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

url = "https://docs.google.com/spreadsheets/d/1Gg3hlqKWxTDNa-MSkNYkUVKG5y_eLfsK9sLFHlhMKhE/export?format=csv"
df = pd.read_csv(url)

df = df.rename(columns={
    "weapon": "Weapon",
    "kills_per_round": "KPR",
    "headshot_rate": "HS",
    "chest_rate": "Chest",
    "leg_rate": "Leg",
    "total_kills": "Total_Kills"
})

df = df.dropna()

for col in ["HS", "Chest", "Leg"]:
    df[col] = df[col].astype(str).str.replace('%', '').astype(float) / 100

df["Total_Kills"] = df["Total_Kills"].astype(str).str.replace(',', '').astype(float)
df["KPR"] = pd.to_numeric(df["KPR"])
df["High_Performance"] = (df["KPR"] >= 1.0).astype(int)

X = df[["KPR", "HS", "Chest", "Leg"]]
y = df["High_Performance"]

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

k = 3
model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"=== KNN (k={k}) — Classificação de Desempenho ===")
print(f"Acurácia: {accuracy_score(y_test, y_pred):.2%}")
print()
print(classification_report(y_test, y_pred))