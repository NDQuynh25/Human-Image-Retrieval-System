from flask import Blueprint, request, jsonify

# Use relative imports
from ..services.image_service import search_image, save_image_data
import tempfile
import traceback
image_routes = Blueprint('image_routes', __name__)


@image_routes.route("/test", methods=["GET"])
def test():
    """
    Test endpoint for image routes
    """
    try:
        # Return a simple test message
        return jsonify({
            "status": "success",
            "message": "Image routes test endpoint is working"
        }), 200
    except Exception as e:
        return jsonify({
            "status": "error", 
            "message": str(e)
        }), 500


@image_routes.route("/upload-image", methods=["POST"])
def upload_image_route():
    """
    Endpoint for uploading image data
    """
    try:
        if 'image' not in request.files:
            print("Không có ảnh được tải lên")
            return jsonify({'error': 'Chưa có ảnh được tải lên'}), 400

        image_file = request.files.getlist('image')[0]
        print(f"Đã nhận file: {image_file.filename}")
        
        # Lưu ảnh tạm thời
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
                image_file.save(temp.name)
                image_path = temp.name
                print(f"Đã lưu ảnh tạm vào: {image_path}")
        except Exception as e:
            print(f"Lỗi khi lưu ảnh tạm: {str(e)}")
            print(traceback.format_exc())
            return jsonify({'error': f'Lỗi khi lưu ảnh tạm: {str(e)}'}), 500

        save_image_data(image_path)
        
        
        return jsonify({
            "status": "success",
            "message": "Image data saved successfully!"
        }), 200
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({
            "status": "error", 
            "message": e
        }), 500
    



@image_routes.route('/search', methods=['POST'])
def search_image_route():
    try:
       
        
        if 'image' not in request.files:
            print("Không có ảnh được tải lên")
            return jsonify({'error': 'Chưa có ảnh được tải lên'}), 400

        image_file = request.files.getlist('image')[0]
        print(f"Đã nhận file: {image_file.filename}")
        
        # Lưu ảnh tạm thời
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
                image_file.save(temp.name)
                image_path = temp.name
                print(f"Đã lưu ảnh tạm vào: {image_path}")
        except Exception as e:
            print(f"Lỗi khi lưu ảnh tạm: {str(e)}")
            return jsonify({'error': f'Lỗi khi lưu ảnh tạm: {str(e)}'}), 500
        
        # Trích xuất đặc trưng từ ảnh
        try:
            print(f"Bắt đầu trích xuất đặc trưng từ ảnh: {image_path}")

            result = search_image(image_path)
            
          
            
            if result is None:
                print("Không thể trích xuất đặc trưng từ ảnh")
                return jsonify({'error': 'Không thể trích xuất đặc trưng từ ảnh'}), 500
            
            return jsonify({'message': 'Đã nhận ảnh'}), 200
        except Exception as e:
            print(f"Lỗi khi trích xuất đặc trưng: {str(e)}")
            print(traceback.format_exc())
            return jsonify({'error': f'Lỗi khi trích xuất đặc trưng: {str(e)}'}), 500
            
    except Exception as e:
        print(f"Lỗi không xác định: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            "status": "error",
            "message": f"Có lỗi xảy ra khi xử lý ảnh: {str(e)}"
        }), 500
