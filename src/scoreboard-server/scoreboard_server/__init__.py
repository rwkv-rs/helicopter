"""FastAPI/PostgreSQL scoreboard service for Helicopter."""

__all__ = ["ScoreboardStore", "create_app"]


def __getattr__(name: str):
    if name == "ScoreboardStore":
        from .db.repository import ScoreboardStore

        return ScoreboardStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def create_app(*args, **kwargs):
    from .application import create_app as _create_app

    return _create_app(*args, **kwargs)
