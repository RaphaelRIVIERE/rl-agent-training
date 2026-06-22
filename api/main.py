import os
import sys
import numpy.core
import numpy.core.numeric
import numpy.core.multiarray
import numpy.core.umath
import numpy.core.fromnumeric
from contextlib import asynccontextmanager

import numpy as np
import gymnasium as gym
from fastapi import FastAPI
from stable_baselines3 import DQN

from api.routes import router

# --- Compatibilité numpy 1.x / 2.x ---
# Le modèle a été entraîné sur Google Colab (numpy 2.x). SB3 sérialise certains
# objets (espaces, tableaux numpy) via cloudpickle, qui enregistre les chemins
# de modules internes. En numpy 2.x, ces chemins pointent vers numpy._core
# (ex. numpy._core.numeric), un module qui n'existe pas en numpy 1.x.
# Charger le modèle localement lève donc : ModuleNotFoundError: numpy._core.
#
# Solution : enregistrer des alias dans sys.modules pour que numpy._core.X
# pointe vers numpy.core.X (son équivalent en numpy 1.x). Les alias sont
# ajoutés APRÈS l'import de SB3/pandas, car ces librairies sont compilées
# pour numpy 1.x et ne doivent pas voir numpy._core au moment de leur import.
sys.modules.setdefault("numpy._core", numpy.core)
sys.modules.setdefault("numpy._core.numeric", numpy.core.numeric)
sys.modules.setdefault("numpy._core.multiarray", numpy.core.multiarray)
sys.modules.setdefault("numpy._core.umath", numpy.core.umath)
sys.modules.setdefault("numpy._core.fromnumeric", numpy.core.fromnumeric)

MODEL_PATH = os.getenv("MODEL_PATH", "notebook/models/dqn_final.zip")

# --- Incompatibilité de version SB3 ---
# Le modèle a été sauvegardé avec SB3 2.9.0 (Colab), mais l'API tourne avec
# SB3 2.4.1 (dernière version compatible avec torch 2.2.2 sur macOS Intel).
# SB3 2.9 a introduit de nouvelles classes (FloatSchedule, LinearSchedule) et
# un nouveau format de sérialisation pour les espaces d'observation/action.
# SB3 2.4.1 ne connaît pas ces classes et ne peut pas les désérialiser.
#
# Solution : fournir directement les objets via custom_objects. SB3 utilisera
# ces valeurs au lieu de tenter de désérialiser les champs problématiques.
# Les espaces sont ceux de LunarLander-v3 (8 observations continues, 4 actions
# discrètes). Les schedules ne servent qu'à l'entraînement, pas à l'inférence.
CUSTOM_OBJECTS = {
    "observation_space": gym.spaces.Box(
        low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
    ),
    "action_space": gym.spaces.Discrete(4),
    "lr_schedule": lambda _: 1e-4,
    "exploration_schedule": lambda _: 0.05,
}

model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = DQN.load(MODEL_PATH, custom_objects=CUSTOM_OBJECTS)
    yield


app = FastAPI(
    title="AstroDynamics Eagle-1",
    description="API de pilotage automatique LunarLander",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(router)
