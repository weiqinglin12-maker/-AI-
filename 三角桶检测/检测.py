import torch
from ultralytics import YOLO
import argparse
import cv2
import os
import shutil
from pathlib import Path

def detect(save_img=False):
    # 解析命令行参数
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    
    # 初始化输出目录
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out, exist_ok=True)

    # 加载模型
    model = YOLO(weights)
    
    # 推理参数设置
    predict_args = {
        'source': source,
        'conf': opt.conf_thres,
        'iou': opt.iou_thres,
        'imgsz': imgsz,
        'device': opt.device,
        'show': view_img,
        'save': save_img,
        'save_txt': save_txt,
        'save_conf': True,  # 默认保存置信度
        'classes': opt.classes,
        'agnostic_nms': opt.agnostic_nms,
        'augment': opt.augment,
        'vid_stride': 1,
        'max_det': 1000
    }

    # 执行预测
    results = model.predict(**predict_args)

# 自定义后处理（示例）
    for result in results:
    # 结果文件路径处理
        save_path = str(Path(out) / Path(result.path).name)
    
    # 绘制带标注的图像
    plotted_img = result.plot(line_width=3, labels=True, conf=True)
    
    if save_img:
        # 判断是否为图像
        is_image = result.path and os.path.splitext(result.path)[1].lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        if is_image:
            # 保存图像
            cv2.imwrite(save_path, plotted_img)
        else:  # 视频处理
            # 初始化 VideoWriter
            if not hasattr(detect, 'vid_writer'):
                h, w = plotted_img.shape[:2]  # 获取图像高度和宽度
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 编码器
                
                # 生成输出视频文件名
                video_save_path = str(Path(out) / f"output.mp4")
                
                detect.vid_writer = cv2.VideoWriter(
                    video_save_path, fourcc, 30, (w, h))  # 假设帧率为 30 FPS
                if not detect.vid_writer.isOpened():
                    print(f"无法初始化 VideoWriter: {video_save_path}")
                    exit(1)
            
            # 写入每一帧
            detect.vid_writer.write(plotted_img)

# 释放视频资源
if hasattr(detect, 'vid_writer'):
    detect.vid_writer.release()
    print(f"视频已保存: {video_save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, 
                        default=r'D:\桌面文件\工程部AI推进项目\三角桶检测\Cone.pt',
                        help='模型权重路径')
    parser.add_argument('--source', type=str, 
                        default=r"D:\桌面文件\基础资料\AI+视频监测\检测视频\安全帽对比.mp4",
                        help='输入源路径')
    parser.add_argument('--output', type=str, 
                        default='D:/桌面文件/工程部AI推进项目/三角桶检测/output',
                        help='输出目录路径')
    parser.add_argument('--img-size', type=int, 
                        default=640,
                        help='推理尺寸（像素）')
    parser.add_argument('--conf-thres', type=float,
                        default=0.4,
                        help='置信度阈值')
    parser.add_argument('--iou-thres', type=float,
                        default=0.5,
                        help='NMS IoU 阈值')
    parser.add_argument('--device', default='',
                        help='计算设备 (cuda device 0,1,2,3 或 cpu)')
    parser.add_argument('--view-img', action='store_true', 
                        help='是否显示图像')
    parser.add_argument('--save-txt', action='store_true', 
                        help='是否保存标签文件')
    parser.add_argument('--classes', nargs='+', type=int, default=None,
                        help='过滤特定类别')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='是否启用 agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                        help='是否启用数据增强')

    opt = parser.parse_args()
    
    # 自动选择设备
    if opt.device == '' and torch.cuda.is_available():
        opt.device = '0'  # 使用 GPU
    else:
        opt.device = 'cpu'  # 使用 CPU
    
    print(f'配置参数: {opt}')
    detect(save_img=True)