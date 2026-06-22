import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import api.main as state

router = APIRouter()


class ObservationRequest(BaseModel):
    observation: list[float]


@router.get(
    "/health",
    tags=["Health"],
    summary="Vérification de l'état du service",
)
def health():
    return {"status": "ok"}


@router.post(
    "/play",
    tags=["Agent"],
    summary="Prédit l'action pour un état donné",
)
def play(request: ObservationRequest):
    if state.model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    obs = np.array(request.observation, dtype=np.float32)

    if obs.shape != (8,):
        raise HTTPException(
            status_code=422,
            detail=f"L'observation doit contenir 8 valeurs, reçu {obs.shape[0]}",
        )

    # deterministic=True : on désactive l'exploration epsilon-greedy,
    # l'agent joue toujours son meilleur coup
    action, _ = state.model.predict(obs, deterministic=True)
    return {"action": int(action)}
