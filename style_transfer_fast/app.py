from flask import Flask, request, jsonify, send_from_directory
import os
from main import main
from werkzeug.utils import secure_filename
from flask_cors import CORS 

app = Flask(__name__, static_folder='frontend/build', static_url_path='')
CORS(app)  # 跨域請求

app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['STYLE_FOLDER'] = 'static/styles'
app.config['OUTPUT_FOLDER'] = 'static/output'

# 確保目錄存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['STYLE_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

@app.route('/api/styles', methods=['GET'])
def get_styles():
    """獲取所有可用的風格圖片"""
    style_images = os.listdir(app.config['STYLE_FOLDER'])
    styles = []
    for style in style_images:
        name, ext = os.path.splitext(style)
        styles.append({
            'id': name,
            'name': name,
            'filename': style,
            'url': f'/static/styles/{style}'
        })
    return jsonify(styles)

@app.route('/api/transfer', methods=['POST'])
def transfer_style():
    """處理風格轉換"""
    try:
        if 'content_image' not in request.files:
            return jsonify({'error': '未上傳內容圖片'}), 400
            
        content_image = request.files['content_image']
        style_name = request.form['style_name']
        
        if content_image.filename == '':
            return jsonify({'error': '未選擇內容圖片檔案'}), 400
        
        # 保存上傳的內容圖片
        filename = secure_filename(content_image.filename)
        content_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        content_image.save(content_path)
        
        # 調用風格轉換功能
        result_image_path = main(content_path, style_name)
        
        return jsonify({
            'success': True,
            'content_image': f'/{content_path}',
            'style_name': style_name,
            'result_image': f'/static/{result_image_path}'
        })
    
    except Exception as e:
        import traceback
        print(f"處理上傳時發生錯誤: {str(e)}")
        traceback.print_exc()
        return (jsonify({'error': str(e)}), 500)

# 提供靜態文件
@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

# 在生產環境中，提供 React 前端
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_react(path):
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    app.run(debug=True)