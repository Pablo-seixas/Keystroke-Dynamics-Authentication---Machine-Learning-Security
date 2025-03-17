import time
import pandas as pd
import joblib
from pynput import keyboard

# Carregar o modelo treinado
model = joblib.load("keystroke_model.pkl")

# Lista para armazenar os tempos de digitação
keystroke_data = []

def on_press(key):
    """Captura quando uma tecla é pressionada e armazena o tempo."""
    try:
        key_pressed = key.char
    except AttributeError:
        key_pressed = str(key)
    
    keystroke_data.append({
        'key': key_pressed,
        'time_pressed': time.time()
    })

def on_release(key):
    """Captura quando uma tecla é solta e registra o tempo total de pressionamento."""
    if key == keyboard.Key.esc:
        return False

    try:
        key_released = key.char
    except AttributeError:
        key_released = str(key)

    for data in reversed(keystroke_data):
        if data['key'] == key_released and 'time_released' not in data:
            data['time_released'] = time.time()
            data['hold_time'] = data['time_released'] - data['time_pressed']
            break

# Iniciar teste de digitação
print("Digite um pequeno texto. Pressione ESC para finalizar...")

with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()

# Convertendo para DataFrame
df_test = pd.DataFrame(keystroke_data)

# Verificar se coletou dados suficientes
if df_test.shape[0] < 5:
    print(" Poucos dados coletados! Digite um texto maior.")
    exit()

# Criar feature de tempo entre teclas
df_test['flight_time'] = df_test['time_pressed'].diff()

# Remover a primeira linha pois não tem referência para flight time
df_test = df_test.iloc[1:].reset_index(drop=True)

# Selecionar apenas as features para prever
X_test = df_test[['hold_time', 'flight_time']]

# Fazer a previsão
prediction = model.predict(X_test)

# Mostrar resultado
resultado = f"O modelo prevê que essa digitação pertence ao usuário: {prediction[0]}"

# Salvar em um arquivo novo com UTF-8
with open("prediction_results.txt", "w", encoding="utf-8") as f:
    f.write(resultado)

# Mostrar no terminal também
print(resultado)
print(" Resultado salvo em 'prediction_results.txt'")
