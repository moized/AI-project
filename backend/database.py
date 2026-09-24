from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from core.config import settings
from sqlalchemy import DateTime, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


if settings.database_url.startswith("sqlite:///"):
    sqlite_target = settings.database_url.removeprefix("sqlite:///")
    if sqlite_target != ":memory:":
        Path(sqlite_target).expanduser().resolve().parent.mkdir(
            parents=True,
            exist_ok=True,
        )


class Base(DeclarativeBase):
    pass


engine = create_engine(
    settings.database_url,
    connect_args=(
        {"check_same_thread": False}
        if settings.database_url.startswith("sqlite")
        else {}
    ),
)
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
)


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    query: Mapped[str] = mapped_column(String(8000), index=True)
    answer: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
