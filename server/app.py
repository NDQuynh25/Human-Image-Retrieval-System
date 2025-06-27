from flask import Flask
from flask_cors import CORS
import sys
import os

from .config.db_config import init_db
from .routers.image_routes import image_routes
from .config.cloudinary_config import config_cloudinary

def create_app():
    app = Flask(__name__)

    # CORS
    CORS(app)

    # Kết nối MongoDB từ config
    init_db()

    # Kết nối Cloudinary từ config
    config_cloudinary()
    
    # Đăng ký routes
    app.register_blueprint(image_routes, url_prefix="/api/v1")
   
    return app

if __name__ == "__main__":
   
    
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
   
    
    app = create_app()
    app.run(debug=True, host='localhost', port=5000)