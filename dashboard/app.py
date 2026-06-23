import streamlit as st
import gymnasium as gym
import requests
import pandas as pd
import plotly.graph_objects as go


def play_episode(api_url):
    env = gym.make("LunarLander-v3")

    obs, _ = env.reset()
    step_count = 0
    total_reward = 0
    terminated = False
    truncated = False

    while not terminated and not truncated:
        response = requests.post(api_url + "/play", json={"observation": obs.tolist()})
        action = response.json()["action"]

        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        step_count += 1

    env.close()
    return {"reward": total_reward, "steps": step_count}


st.set_page_config(page_title="Eagle-1 Dashboard", layout="wide")

st.title("Eagle-1 — Tableau de bord")

api_url = st.sidebar.text_input("URL de l'API", value="http://localhost:8000")
n_episodes = st.sidebar.slider("Nombre d'épisodes", min_value=5, max_value=50, value=20)
run = st.sidebar.button("Lancer les épisodes")

if not run:
    st.info("Lance des épisodes via la sidebar pour voir les résultats.")

if run:
    results = []
    progress_bar = st.progress(0)

    for i in range(n_episodes):
        result = play_episode(api_url)
        results.append(result)
        progress_bar.progress((i + 1) / n_episodes)

    df = pd.DataFrame(results)
    df.index = df.index + 1  # épisodes numérotés à partir de 1

    mean_reward = df["reward"].mean()
    std_reward = df["reward"].std()
    best_reward = df["reward"].max()
    success_rate = (df["reward"] >= 200).mean() * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Récompense moyenne", f"{mean_reward:.1f}", f"±{std_reward:.1f}")
    col2.metric("Meilleure récompense", f"{best_reward:.1f}")
    col3.metric("Taux de succès (≥200)", f"{success_rate:.0f}%")

    st.subheader("Récompense par épisode")

    fig = go.Figure()

    # bande ±1 écart-type
    fig.add_trace(go.Scatter(
        x=list(df.index) + list(df.index[::-1]),
        y=list(df["reward"] + std_reward) + list((df["reward"] - std_reward)[::-1]),
        fill="toself",
        fillcolor="rgba(99, 110, 250, 0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        name="±1 écart-type",
        hoverinfo="skip",
    ))

    # courbe des récompenses
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["reward"],
        mode="lines+markers",
        name="Récompense",
        line=dict(color="rgb(99, 110, 250)"),
        marker=dict(size=6),
    ))

    # ligne de moyenne
    fig.add_hline(
        y=mean_reward,
        line_dash="dash",
        line_color="orange",
        annotation_text=f"Moyenne : {mean_reward:.1f}",
        annotation_position="bottom right",
    )

    # ligne objectif
    fig.add_hline(
        y=200,
        line_dash="dot",
        line_color="green",
        annotation_text="Objectif : 200",
        annotation_position="top right",
    )

    fig.update_layout(
        xaxis_title="Épisode",
        yaxis_title="Récompense totale",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400,
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Détail des épisodes")

    seuil = st.slider(
        "Afficher les épisodes avec récompense ≥",
        min_value=int(df["reward"].min()) - 1,
        max_value=int(df["reward"].max()) + 1,
        value=int(df["reward"].min()) - 1,
    )

    df_filtered = df[df["reward"] >= seuil].copy()
    df_filtered.columns = ["Récompense", "Steps"]
    st.dataframe(df_filtered, use_container_width=True)
