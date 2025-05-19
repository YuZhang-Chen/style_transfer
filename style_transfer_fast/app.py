from flask import Flask, render_template, request, redirect, url_for
import os
from main import main
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['STYLE_FOLDER'] = 'static/styles'
app.config['OUTPUT_FOLDER'] = 'static/output'  # 修改為與 main.py 一致的輸出目錄

# 確保上傳目錄存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['STYLE_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)  # 使用更新後的輸出目錄

@app.route('/')
def home():
    # 獲取可用的風格圖片
    style_images = os.listdir(app.config['STYLE_FOLDER'])
    return render_template('index.html', style_images=style_images)

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        content_image = request.files['content_image'] # 獲取上傳的內容圖片
        style_name = request.form['style_image']
        
        if content_image and content_image.filename != '': # 確保有上傳的內容圖片
            # 儲存上傳的內容圖片
            filename = secure_filename(content_image.filename)
            content_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            content_image.save(content_path)
            print(f"內容圖片已保存到: {content_path}")
            
            # 呼叫 main 函數進行風格轉換
            print(f"開始處理風格轉換，風格: {style_name}")
            result_image_path = main(content_path, style_name)
            print(f"風格轉換完成，結果保存於: {result_image_path}")
            
            # 返回結果頁面，顯示轉換結果
            return render_template('result.html', 
                                  result_image=result_image_path,
                                  content_image=os.path.join('uploads', filename),
                                  style_name=style_name)
        else:
            return render_template('error.html', error="未選擇內容圖片或檔案為空")
    
    except Exception as e:
        # 捕獲並記錄任何異常
        import traceback
        print(f"處理上傳時發生錯誤: {str(e)}")
        traceback.print_exc()
        return render_template('error.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)