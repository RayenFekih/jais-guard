from pydantic import BaseModel, ConfigDict


class Config(BaseModel):
    """Application settings. Immutable after construction."""

    model_config = ConfigDict(frozen=True)

    model_id: str = "IFM/Jais-3-0.9B"
    max_new_tokens: int = 256
    prompt: str = "ما هي عاصمة تونس؟"


config = Config()
