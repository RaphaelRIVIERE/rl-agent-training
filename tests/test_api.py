import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    # On remplace DQN.load par un faux modèle 
    mock_model = MagicMock()
    mock_model.predict.return_value = (np.int64(2), None)

    with patch("api.main.DQN") as MockDQN:
        MockDQN.load.return_value = mock_model
        from api.main import app
        with TestClient(app) as c:
            yield c


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_play_valid_observation(client):
    obs = [0.0] * 8
    response = client.post("/play", json={"observation": obs})
    assert response.status_code == 200
    data = response.json()
    assert data["action"] == 2


def test_play_invalid_observation(client):
    obs = [0.0] * 5
    response = client.post("/play", json={"observation": obs})
    assert response.status_code == 422
