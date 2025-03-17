import time
import pandas as pd
from pynput import keyboard

# Lista para armazenar os tempos de digitação
keystroke_data = []

def on_press(key):
    """Captura quando uma tecla é pressionada e armazena o tempo."""
    try:
        key_pressed = key.char  # Captura o caractere digitado
    except AttributeError:
        key_pressed = str(key)  # Para teclas especiais como Shift, Enter, etc.
    
    keystroke_data.append({
        'key': key_pressed,
        'time_pressed': time.time()
    })

def on_release(key):
    """Captura quando uma tecla é solta e registra o tempo total de pressionamento."""
    if key == keyboard.Key.esc:
        # Se pressionar ESC, para de gravar
        return False

    try:
        key_released = key.char
    except AttributeError:
        key_released = str(key)
    
    # Encontrando a tecla pressionada mais recente correspondente
    for data in reversed(keystroke_data):
        if data['key'] == key_released and 'time_released' not in data:
            data['time_released'] = time.time()
            data['hold_time'] = data['time_released'] - data['time_pressed']
            break

# Iniciar coleta de digitação
print("Digite um pequeno texto. Pressione ESC para finalizar...")

with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()

# Convertendo para DataFrame
df_keystroke = pd.DataFrame(keystroke_data)

# Limpando os dados e mantendo apenas as colunas necessárias
df_keystroke = df_keystroke.dropna().reset_index(drop=True)

# Criando feature de tempo entre teclas (flight time)
df_keystroke['flight_time'] = df_keystroke['time_pressed'].diff()

# Removendo a primeira linha pois não tem referência para flight time
df_keystroke = df_keystroke.iloc[1:].reset_index(drop=True)

# Salvando os dados em um arquivo CSV
df_keystroke.to_csv("keystroke_data.csv", index=False)

# Exibindo os primeiros dados coletados
print(df_keystroke.head())
