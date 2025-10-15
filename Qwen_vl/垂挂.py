# -*- coding: utf-8 -*-
import os
import base64
import requests
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_fixed

# API请求的URL和头部信息
api_url = "http://10.32.41.228:39080/al/chat/completions"
headers = {
    "Content-Type": "application/json;charset=UTF-8",
    "AI-API-CODE": "xfURehmUxu",
    "AI-APP-KEY": "F47CAzybzG"
}

# 图片所在的文件夹路径
folder_path = r"D:\桌面文件\工程部AI推进项目\Qwen_vl\垂挂-测试"

# 存储结果的列表
results = []
failed_files = []

@retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
def process_image_with_retry(file_path):
    """处理单个图片文件并发送API请求（带重试机制）"""
    try:
        with open(file_path, 'rb') as img:
            # 将图片转换为Base64编码
            img_data = img.read()
            img_base64 = base64.b64encode(img_data).decode('utf-8')
            
            # 根据图片扩展名确定MIME类型
            ext = os.path.splitext(file_path)[1].lower()[1:]
            mime_type = f"image/{ext}" if ext in ['jpeg', 'jpg', 'png', 'gif', 'bmp'] else "image/jpeg"
            
            image_url = f"data:{mime_type};base64,{img_base64}"
            
            payload = {
                "model": "qwen2-vl-72b-aqw",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "请根据以下几点对图片中的光缆状态进行分析:识别预留线圈：预留线圈通常呈现出规则的圆形或椭圆形，位置固定且不随意摆动。 - 注意预留线圈一般位于设计预定的位置，不会对周围环境造成阻碍或危险。判断垂挂特征： - 弧形向下：垂挂的光缆会形成明显的向下弧度。 - 大曲率半径：检查光缆的弯曲部分，若其弧度较大且松弛，则可能是垂挂现象。 - 垂挂的光缆通常表现为无序和不受控制的状态，而非紧绷的布局。 预留线圈虽然也可能有弧度，但形状较为规则；而垂挂的光缆则显得更加松散和不规则。 请特别关注上述特征，判断图片中的图像是否垂挂，回答格式：垂挂/不垂挂，理由。。。"},
                            {"type": "image_url", "image_url": {"url": image_url}}
                        ]
                    }
                ],
                "max_tokens": 600,
                "stream": False,
                "temperature": 0.1,
                "top_p": 0.95
            }
            
            response = requests.post(api_url, headers=headers, json=payload, timeout=30)
            
            # 检查响应状态码
            if response.status_code != 200:
                print(f"Processing {file_path}:")
                print(f"Error: Received status code {response.status_code}")
                print(f"Response content: {response.text}")
                return None
            
            # 尝试解析JSON响应
            try:
                response_json = response.json()
                # 提取content字段作为状态
                status = response_json.get('choices', [{}])[0].get('message', {}).get('content', '未知')
                print(f"Processing {file_path}:")
                print(response_json)
                return {"图片": file_path, "状态": status}
            except ValueError:
                print(f"Processing {file_path}:")
                print("Error: Response is not valid JSON")
                print(f"Response content: {response.text}")
                return None
                
    except requests.exceptions.RequestException as e:
        print(f"Processing {file_path}:")
        print(f"Error: {e}")
        return None

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):  # 检查是否为图片文件
        result = process_image_with_retry(file_path)
        if result is None:
            failed_files.append(file_path)
        else:
            results.append(result)

# 处理失败的文件，直到所有文件都成功或达到最大重试次数
while failed_files:
    new_failed_files = []
    for file_path in failed_files:
        result = process_image_with_retry(file_path)
        if result is None:
            new_failed_files.append(file_path)
        else:
            results.append(result)
    failed_files = new_failed_files
# 将结果保存到Excel文件
output_file = r"D:\桌面文件\工程部AI推进项目\Qwen_vl\results.xlsx"
df = pd.DataFrame(results)
df.to_excel(output_file, index=False, sheet_name='Results')

print(f"Results saved to {output_file}")
