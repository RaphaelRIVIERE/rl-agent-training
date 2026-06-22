import gymnasium as gym
import requests
import streamlit as st


env = gym.make("LunarLander-v3", render_mode="rgb_array")

obs, _ = env.reset()

total_reward = 0
terminated = False
truncated = False

frame_placeholder = st.empty()


while not terminated and not truncated:
    frame = env.render()
    frame_placeholder.image(frame)

    response = requests.post(
        "http://localhost:8000/play",
        json={"observation": obs.tolist()}
    )
    action = response.json()["action"]

    obs, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward

env.close()
st.write(f"Récompense totale : {total_reward:.2f}")
