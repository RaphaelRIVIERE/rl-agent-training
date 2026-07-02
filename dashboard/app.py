import streamlit as st
import gymnasium as gym
import requests
import pandas as pd
import plotly.graph_objects as go
import numpy as np


def play_episode(api_url):
    env = gym.make("LunarLander-v3")

    obs, _ = env.reset()
    initial_obs = obs.copy()
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
    return {"reward": total_reward, "steps": step_count, "initial_obs": initial_obs}


st.set_page_config(page_title="Eagle-1 Dashboard", layout="wide")

st.title("Tableau de bord de Eagle-1 ")

api_url = st.sidebar.text_input("URL de l'API", value="http://localhost:8000")
n_episodes = st.sidebar.slider("Nombre d'épisodes", min_value=5, max_value=50, value=20)
run = st.sidebar.button("Lancer les épisodes")

if run:
    results = []
    progress_bar = st.progress(0)

    for i in range(n_episodes):
        result = play_episode(api_url)
        results.append(result)
        progress_bar.progress((i + 1) / n_episodes)

    df = pd.DataFrame(results)
    df.index = df.index + 1  # épisodes numérotés à partir de 1

    st.session_state["df"] = df
    st.session_state["results"] = results

if "df" not in st.session_state:
    st.info("Lance des épisodes via la sidebar pour voir les résultats.")
else:
    df = st.session_state["df"]
    results = st.session_state["results"]

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

    df_filtered = df[df["reward"] >= seuil][["reward", "steps"]].copy()
    df_filtered.columns = ["Récompense", "Steps"]
    st.dataframe(df_filtered, use_container_width=True)

    st.subheader("Diversité des conditions de départ")
    st.caption(
        "Vérifie que l'agent n'a pas mémorisé des scénarios spécifiques : "
        "si les états initiaux sont variés et que les performances restent bonnes, il n'y a pas de surapprentissage."
    )

    initial_obs_matrix = np.array([r["initial_obs"] for r in results])

    # Distances euclidiennes entre toutes les paires (sur les 6 premières dimensions : position, vitesse, angle)
    similarity_threshold = st.slider(
        "Seuil de similarité (distance euclidienne)",
        min_value=0.1,
        max_value=2.0,
        value=0.5,
        step=0.1,
    )

    n = len(initial_obs_matrix)
    similar_pairs = 0
    total_pairs = n * (n - 1) // 2
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(initial_obs_matrix[i, :6] - initial_obs_matrix[j, :6])
            if dist < similarity_threshold:
                similar_pairs += 1

    col_a, col_b = st.columns(2)
    col_a.metric("Paires d'épisodes similaires", f"{similar_pairs} / {total_pairs}")
    col_b.metric(
        "Taux de similarité",
        f"{similar_pairs / total_pairs * 100:.1f}%",
        help="Un taux faible confirme la diversité des scénarios testés.",
    )

    fig_div = go.Figure()
    fig_div.add_trace(go.Scatter(
        x=initial_obs_matrix[:, 0],
        y=initial_obs_matrix[:, 1],
        mode="markers",
        marker=dict(
            size=10,
            color=df["reward"],
            colorscale="RdYlGn",
            colorbar=dict(title="Récompense"),
            showscale=True,
        ),
        text=[f"Épisode {i+1}<br>Récompense : {r:.1f}" for i, r in enumerate(df["reward"])],
        hovertemplate="%{text}<br>x₀=%{x:.3f}, y₀=%{y:.3f}<extra></extra>",
    ))
    fig_div.update_layout(
        xaxis_title="Position horizontale initiale (x₀)",
        yaxis_title="Position verticale initiale (y₀)",
        height=400,
    )
    st.plotly_chart(fig_div, use_container_width=True)
