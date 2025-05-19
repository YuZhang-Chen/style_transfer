import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os

base_image_path = "./images/img.jpg"
style_folder_path = "./styles"
style_images = [os.path.join(style_folder_path, f) for f in os.listdir(style_folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

original_width, original_height = keras.utils.load_img(
    base_image_path
).size
image_height = 300
image_width = int(original_width * image_height / original_height)


# 使用Keras載入並預處理圖片，轉換為神經網路可接受的格式
def process_image(image_path):
    img = keras.utils.load_img(image_path, target_size=(image_height, image_width))
    img = keras.utils.img_to_array(img)  # 轉為numpy數組
    img = np.expand_dims(img, axis=0)  # 添加維度
    img = keras.applications.vgg19.preprocess_input(img)  # VGG19預處理：減去RGB均值標準化輸入
    return img

# 將神經網路輸出的張量轉換回可視化的圖片格式
def deprocess_image(img):
    img = img.reshape((image_height, image_width, 3))
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68
    img = img[:, :, ::-1]
    img = np.clip(img, 0, 255).astype("uint8")
    return img

# 載入VGG19模型做特徵提取
model = keras.applications.VGG19(
    include_top=False,  # 不包含分類器層，只使用卷積部分
    weights="imagenet",  # 使用ImageNet預訓練權重
)

# 創建特徵提取模型，用於從不同層獲取特徵
output_dict = dict([(layer.name, layer.output) for layer in model.layers])
feature_extractor = keras.Model(inputs=model.inputs, outputs=output_dict)


# 計算內容損失：測量在高級特徵上的均方誤差
def content_loss(base_img, combination_img):
    return tf.reduce_sum(tf.square(combination_img - base_img))


# 計算Gram矩陣：表示特徵之間的相關性，用於風格表示
def gram_matrix(x):
    x = tf.transpose(x, (2, 0, 1))  # 重新排列維度
    features = tf.reshape(x, (tf.shape(x)[0], -1))  # 將特徵展開
    gram = tf.matmul(features, tf.transpose(features))  # 計算特徵的內積
    return gram

# 計算風格損失：測量生成圖像與風格圖像在Gram矩陣上的差異
def style_loss(style, combination_img):
    S = gram_matrix(style)
    C = gram_matrix(combination_img)
    channels = 3
    size = image_height * image_width
    return tf.reduce_sum(tf.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))


# 計算總變差損失
def total_variation_loss(x):
    a = tf.square(
        x[:, :image_height-1, :image_width-1, :] - x[:, 1:, : image_width-1, :]
    )
    b = tf.square(
        x[:, :image_height-1, :image_width-1, :] - x[:, :image_height-1, 1:, :]
    )
    return tf.reduce_sum(tf.pow(a+b, 1.25))  # 1.25 is a hyperparameter

# VGG19模型中用於提取風格的卷積層
style_layers_names = [
    "block1_conv1",
    "block2_conv1",
    "block3_conv1",
    "block4_conv1",
    "block5_conv1",
]


content_layer_name = "block5_conv2"
total_variation_weight = 1e-4
style_weight = 1.0
content_weight = 1e-3

# 計算總損失函數：結合內容損失、風格損失和變分損失
def compute_loss(combination_image, base_image, style_reference_image):
    # 將三張圖像合併成一個批次輸入到模型
    input_tensor = tf.concat([base_image, style_reference_image, combination_image], axis=0)
    features = feature_extractor(input_tensor)  # 提取所有層的特徵

    loss = tf.zeros(shape=())

    layer_features = features[content_layer_name]
    base_image_features = layer_features[0, :, :, :]
    combination_features = layer_features[2, :, :, :]
    loss += content_weight * content_loss(base_image_features, combination_features)

    for layer_name in style_layers_names:
        layer_features = features[layer_name]
        style_reference_features = layer_features[1, :, :, :]
        combination_features = layer_features[2, :, :, :]
        style_loss_value = style_loss(style_reference_features, combination_features)
        loss += (style_weight / len(style_layers_names)) * style_loss_value

    loss += total_variation_weight * total_variation_loss(combination_image)
    return loss


@tf.function  # TensorFlow函數裝飾器：將Python函數轉換為高效的TensorFlow圖執行
def compute_loss_and_grads(combination_image, base_image, style_reference_image):
    # 使用自動微分計算損失和梯度
    with tf.GradientTape() as tape:  # 梯度帶：自動記錄操作以便反向傳播
        tape.watch(combination_image)  # 追蹤需要計算梯度的變量
        loss = compute_loss(combination_image, base_image, style_reference_image)
    grads = tape.gradient(loss, combination_image)  # 計算梯度
    
    # 梯度裁剪：防止梯度爆炸問題
    grads = tf.clip_by_norm(grads, 5.0)
    
    return loss, grads

def train_style_transfer(base_image_path, style_image_path, output_prefix, iterations=1000):
    base_image = process_image(base_image_path)
    style_reference_image = process_image(style_image_path)
    combination_image = tf.Variable(process_image(base_image_path))
    
    # 使用Adam優化器進行梯度下降，自適應調整學習率
    optimizer = keras.optimizers.Adam(
        learning_rate=0.05
    )
    
    # 儲存損失歷史
    loss_history = []

    for i in range(1, iterations+1):
        # 計算損失和梯度
        loss, grads = compute_loss_and_grads(
            combination_image, base_image, style_reference_image
        )
        optimizer.apply_gradients([(grads, combination_image)])
        loss_history.append(float(loss))

        if i % 100 == 0:
            print(f"Style: {os.path.basename(style_image_path)}, Iteration {i}, loss: {loss:.2f}")
            img = deprocess_image(combination_image.numpy())
            fname = f"{output_prefix}_iteration_{i}.png"
            keras.utils.save_img(fname, img)
            print(f"Image saved as {fname}")
    
    # 保存最終結果
    img = deprocess_image(combination_image.numpy())
    fname = f"{output_prefix}_final.png"
    keras.utils.save_img(fname, img)
    
    return loss_history


os.makedirs("results", exist_ok=True)

# 儲存所有風格的損失歷史
all_loss_histories = {}

# 為每個風格圖片訓練模型
for style_path in style_images:
    style_name = os.path.splitext(os.path.basename(style_path))[0]
    output_prefix = f"results/{style_name}"
    
    print(f"\n\n開始轉換風格: {style_name}\n")
    loss_history = train_style_transfer(base_image_path, style_path, output_prefix, iterations=1000)
    
    all_loss_histories[style_name] = loss_history

# 繪製所有風格的損失圖
plt.figure(figsize=(12, 8))
for style_name, loss_history in all_loss_histories.items():
    plt.plot(loss_history, label=style_name)

plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Style Transfer Training Loss')
plt.legend()
plt.grid(True)
plt.yscale('log')  # 使用對數刻度更容易觀察損失下降
plt.savefig('results/loss_comparison.png')
plt.show()

# 為每個風格分別繪製損失圖
for style_name, loss_history in all_loss_histories.items():
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title(f'Style Transfer Loss: {style_name}')
    plt.grid(True)
    plt.yscale('log')
    plt.savefig(f'results/{style_name}_loss.png')
    plt.close()