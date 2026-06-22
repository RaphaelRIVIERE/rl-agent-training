from fastapi import APIRouter

router = APIRouter()


@router.get(
    "/health",
    tags=["Health"],
    summary="Vérification de l'état du service",
)
def health():
    return {"status": "ok"}
