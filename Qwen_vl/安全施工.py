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
folder_path = r"D:\桌面文件\工程部AI推进项目\Qwen_vl\安全施工-c测试"

# 存储结果的列表
results = []
failed_files = []

@retry(stop=stop_after_attempt(4), wait=wait_fixed(5))
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
            

            # 构建API请求的payload
            payload = {
    "model": "qwen2-vl-72b-aqw",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": """
评估图片中的登高作业场景是否符合所有安全规定，请使用以下提示词进行判断：
保护扶梯：确认是否有其他工作人员在地面扶稳梯子，即图片中有两人作业。
警示围挡：检查施工区域周围是否设置了明显的警示标志（如交通锥、警示带或警示牌），以防止无关人员误入危险区。
佩戴安全帽：确保每位作业人员都佩戴了安全帽，以保护头部免受可能的伤害。
穿反光衣：确认所有作业人员都穿着了反光衣，提高可见度，尤其是在光线不足的环境下工作时。
请关注上述特征，判断图片中的图像是否安全操作，回答格式：安全/不安全，理由：有。。。/没有。。。。
"""
                },
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
output_file = r"D:\桌面文件\工程部AI推进项目\Qwen_vl\safe.xlsx"
df = pd.DataFrame(results)
df.to_excel(output_file, index=False, sheet_name='Results')

print(f"Results saved to {output_file}")