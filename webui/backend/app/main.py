"""FastAPI app entrypoint for the Mimosa Observatory backend."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .live import hub
from .routes import router
from .settings import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    hub.start()
    try:
        yield
    finally:
        await hub.stop()


app = FastAPI(title="Mimosa Observatory", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
def index() -> dict[str, str]:
    return {"service": "mimosa-observatory", "docs": "/docs", "api": "/api"}
