from flask import Flask
from app.routes.main import main_bp


def create_app():
    app = Flask(__name__)

    # Blueprint 등록
    app.register_blueprint(main_bp, url_prefix='/api')

    return app