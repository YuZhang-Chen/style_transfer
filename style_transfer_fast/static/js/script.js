document.addEventListener('DOMContentLoaded', function() {
    // 處理內容圖片上傳與預覽
    const contentInput = document.getElementById('content_image');
    const contentPreview = document.getElementById('content-preview');
    
    if (contentInput) {
        contentInput.addEventListener('change', function() {
            const file = this.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    contentPreview.innerHTML = '';
                    const img = document.createElement('img');
                    img.src = e.target.result;
                    img.style.maxWidth = '100%';
                    img.style.maxHeight = '100%';
                    contentPreview.appendChild(img);
                }
                reader.readAsDataURL(file);
            }
        });
    }
    
    // 處理風格圖片選擇與預覽
    const styleSelect = document.getElementById('style-select');
    const stylePreview = document.getElementById('style-preview');
    
    if (styleSelect) {
        styleSelect.addEventListener('change', function() {
            const selectedStyle = this.value;
            if (selectedStyle) {
                                
                stylePreview.innerHTML = '';
                const img = document.createElement('img');
                img.src = `../../static/styles/${selectedStyle}.jpg`;
                
                img.style.maxWidth = '100%';
                img.style.maxHeight = '100%';
                img.alt = selectedStyle;
                img.onerror = function() {
                    stylePreview.innerHTML = '<p>無法載入風格圖片</p>';
                };
                stylePreview.appendChild(img);
            }
        });
    }
});
