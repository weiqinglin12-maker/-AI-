from flask import Flask, render_template, request, Response
import os
import base64
from openai import OpenAI
from dotenv import load_dotenv
import threading
from queue import Queue

load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB限制
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 异步任务队列
task_queue = Queue()

def process_video(video_path, prompt):
    client = OpenAI(
        api_key="sk-b43b2c78c4684887a0822b4091f34367",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    
    # 分块读取并编码视频文件
    def encode_video_in_chunks(video_path, chunk_size=5 * 1024 * 1024):  # 每块 5MB
        with open(video_path, "rb") as video_file:
            while chunk := video_file.read(chunk_size):
                yield base64.b64encode(chunk).decode("utf-8")
    
    # 获取视频 MIME 类型
    file_ext = video_path.split('.')[-1].lower()
    mime_type = f"video/{file_ext}" if file_ext in ['mp4', 'mov', 'avi'] else "video/mp4"

    # 创建 API 请求
    encoded_chunks = list(encode_video_in_chunks(video_path))
    completion = client.chat.completions.create(
        model="qvq-max",
        messages=[
            {"role": "system", "content": [{"type": "text", "text": "你是一个专业的视频内容分析师。"}]},
            {"role": "user", "content": [
                {"type": "video_url", "video_url": {"url": f"data:{mime_type};base64,{encoded_chunks[0]}"}},
                {"type": "text", "text": prompt},
            ]}
        ],
        stream=True,
    )
    return completion

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    # 验证文件存在
    if 'video' not in request.files:
        return {"error": "请选择视频文件"}, 400
    
    file = request.files['video']
    prompt = request.form.get('prompt', '这段视频讲的是什么内容？')
    
    # 验证文件名有效性
    if file.filename == '' or not allowed_file(file.filename):
        return {"error": "不支持的文件格式"}, 400

    # 保存临时文件
    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(temp_path)

    # 使用线程池异步处理任务
    def background_task():
        try:
            completion = process_video(temp_path, prompt)
            is_answering = False
            
            for chunk in completion:
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    
                    # 处理思考过程
                    if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                        content = delta.reasoning_content.replace('\n', '\\n')
                        task_queue.put(f"data: {{\"type\":\"reasoning\",\"content\":\"{content}\"}}\n\n")
                    
                    # 处理正式回答
                    elif delta.content:
                        if not is_answering:
                            task_queue.put("data: {\"type\":\"answer_start\"}\n\n")
                            is_answering = True
                        content = delta.content.replace('\n', '\\n').replace('"', '\\"')
                        task_queue.put(f"data: {{\"type\":\"answer\",\"content\":\"{content}\"}}\n\n")
            
        finally:
            # 清理临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)
            task_queue.put(None)  # 标记任务结束

    # 启动后台任务
    threading.Thread(target=background_task).start()

    # 流式返回结果
    def generate():
        while True:
            item = task_queue.get()
            if item is None:  # 任务结束
                break
            yield item

    return Response(generate(), mimetype='text/event-stream')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'mp4', 'mov', 'avi', 'mkv'}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)