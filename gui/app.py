import gymnasium as gym
import requests
import streamlit as st

st.set_page_config(page_title="Eagle-1 — GUI", layout="centered")
st.title("Eagle-1 — Visualisation d'un épisode")

api_url = st.sidebar.text_input("URL de l'API", value="http://localhost:8000")

if not st.button("Lancer un épisode"):
    st.info("Clique sur le bouton pour visualiser un épisode joué par l'agent.")
    st.stop()

# Vérification que l'API est disponible avant de créer l'environnement
try:
    requests.get(api_url + "/health", timeout=2).raise_for_status()
except Exception:
    st.error(f"Impossible de contacter l'API sur {api_url}. Lance d'abord : uvicorn api.main:app")
    st.stop()

env = gym.make("LunarLander-v3", render_mode="rgb_array")
obs, _ = env.reset()

total_reward = 0.0
step_count = 0
terminated = False
truncated = False

frame_placeholder = st.empty()
status_placeholder = st.empty()

while not terminated and not truncated:
    frame = env.render()
    frame_placeholder.image(frame, caption=f"Step {step_count} | Récompense cumulée : {total_reward:.2f}")

    try:
        response = requests.post(
            api_url + "/play",
            json={"observation": obs.tolist()},
            timeout=5,
        )
        response.raise_for_status()
        action = response.json()["action"]
    except Exception as e:
        env.close()
        st.error(f"Erreur lors de l'appel à l'API : {e}")
        st.stop()

    obs, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward
    step_count += 1

env.close()

if total_reward >= 200:
    st.success(f"Atterrissage réussi en {step_count} steps — Récompense : {total_reward:.2f}")
else:
    st.warning(f"Épisode terminé en {step_count} steps — Récompense : {total_reward:.2f}")
