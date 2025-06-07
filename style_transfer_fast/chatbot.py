import os
from openai import AzureOpenAI
from dotenv import load_dotenv

endpoint = "https://c1121-mbli76ld-eastus2.cognitiveservices.azure.com/"
model_name = "gpt-4o"
deployment = "gpt-4o"

# Load environment variables from .env file

# Load the environment variables
load_dotenv()

subscription_key = os.getenv("APIKEY")
api_version = "2024-12-01-preview"

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

def chatbot(content_image_url, style_image_url, output_image_url):
    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "你是一位懂圖像風格轉換的AI助教，善於用簡單語言教學。",
            },
            {
                "role": "user",
                "content": f"""
    這是神經風格轉換的三張圖像：

    1. 內容圖：{content_image_url}
    2. 風格圖：{style_image_url}
    3. 結果圖：{output_image_url}

    請用簡易、親切的方式解釋這三張圖：
    - 結果圖保留了哪些內容圖的資訊？
    - 套用了哪些風格圖的視覺效果？
    - 整體風格轉換成功嗎？
    - 請給予學生情緒價值
    """
            }
        ],
        max_tokens=1024,
        temperature=0.8,
        top_p=1.0,
        model=deployment
    )

    res = (response.choices[0].message.content)
    return res

# url = "http://127.0.0.1:5000"
# content_image_url = f"{url}/uploads/img_0000.jpg"
# style_image_url = f"{url}/styles/Night.jpg"
# output_image_url = f"{url}/output/stylized_img_0000_Night.jpg"
# print(content_image_url)
# response = chatbot(content_image_url,style_image_url,output_image_url)

# print(response)