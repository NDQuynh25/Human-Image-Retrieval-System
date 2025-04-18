from flask import Flask
from flask_cors import CORS

# Use relative imports
from .config.db_config import init_db
from .routers.image_routes import image_routes

def create_app():
    app = Flask(__name__)

    # CORS
    CORS(app)

    # Kết nối MongoDB từ config
    init_db()
    
    # Đăng ký routes
    app.register_blueprint(image_routes, url_prefix="/api/v1")
   
    return app

if __name__ == "__main__":
    # When running directly, we need to use absolute imports
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from server.config.db_config import init_db
    from server.routers.image_routes import image_routes
    
    app = create_app()
    app.run(debug=True, host='localhost', port=5000)