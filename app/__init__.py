import logging
import os
from flask import Flask
from dotenv import load_dotenv

load_dotenv()


def create_app():
    app = Flask(
        __name__,
        template_folder="../templates",
        static_folder="../static",
    )

    app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "dev-secret-change-in-production")
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload limit
    app.config["MODEL_DIR"] = os.getenv("MODEL_DIR", "models")
    app.config["DATA_DIR"] = os.getenv("DATA_DIR", "data")

    _configure_logging(app)
    _ensure_directories(app)

    from app.routes import bp
    app.register_blueprint(bp)

    return app


def _configure_logging(app):
    log_level = logging.DEBUG if os.getenv("FLASK_ENV") == "development" else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    app.logger.setLevel(log_level)


def _ensure_directories(app):
    for directory in [app.config["MODEL_DIR"], app.config["DATA_DIR"]]:
        os.makedirs(directory, exist_ok=True)
