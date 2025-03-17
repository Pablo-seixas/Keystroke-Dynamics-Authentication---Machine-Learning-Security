import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 🔹 Carregar os dados coletados
df = pd.read_csv("keystroke_data.csv")

# 🔹 Verificar se o arquivo contém dados suficientes
if df.shape[0] < 5:  # Agora permite rodar com 5 ou mais linhas
    print("⚠️ Poucos dados coletados, o modelo pode ter baixa precisão!")


# 🔹 Criar identificadores fictícios para os usuários (para fins de teste)
df["user_id"] = np.random.randint(0, 3, size=len(df))  # Simulando 3 usuários diferentes

# 🔹 Selecionar as features relevantes para o modelo
features = ["hold_time", "flight_time"]
X = df[features]
y = df["user_id"]

# 🔹 Dividir os dados para treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 Criar e treinar o modelo de Machine Learning
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 🔹 Fazer previsões e calcular a precisão
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# 🔹 Exibir resultados
print(f"✅ Modelo treinado com sucesso! Precisão: {accuracy:.2%}")

# 🔹 Salvar o modelo treinado
import joblib
joblib.dump(model, "keystroke_model.pkl")

print("✅ Modelo salvo como 'keystroke_model.pkl'")

