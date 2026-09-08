from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    data_dir: Path = Path("/data")
    hackathon_dir: Path = Path("/hackathon")
    tracing_db_path: Path = Path("/data/tracing.db")
    log_level: str = "INFO"

    gigachat_credentials: str = ""
    gigachat_scope: str = "GIGACHAT_API_PERS"
    gigachat_model: str = "GigaChat-Max"
    gigachat_verify_ssl: bool = False


settings = Settings()
