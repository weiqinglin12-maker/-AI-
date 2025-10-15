import argparse
import os
import platform
import shutil
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

# 导入YOLOv5的模块和工具函数
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, plot_one_box, strip_optimizer)
from utils.torch_utils import select_device, load_classifier, time_synchronized


def detect(save_img=False):
    # 解析命令行参数
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # 初始化
    device = select_device(opt.device)  # 选择设备（CPU或GPU）
    if os.path.exists(out):
        shutil.rmtree(out)  # 删除输出文件夹
    os.makedirs(out)  # 创建新的输出文件夹
    half = device.type != 'cpu'  # 只有在CUDA上才使用半精度

    # 加载模型
    model = attempt_load(weights, map_location=device)  # 加载FP32模型
    imgsz = check_img_size(imgsz, s=model.stride.max())  # 检查图片大小是否符合要求
    if half:
        model.half()  # 转换为FP16

    # 第二阶段分类器（可选）
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # 初始化
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # 加载权重
        modelc.to(device).eval()

    # 设置数据加载器
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # 如果是摄像头输入，开启以加快推理速度
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # 获取类别名称和颜色
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # 进行推理
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # 初始化图像
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # 运行一次以预热模型
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8转fp16/32
        img /= 255.0  # 归一化到0-1
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # 推理
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # 应用非极大值抑制(NMS)
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # 应用分类器（如果启用）
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # 处理每个检测结果
        for i, det in enumerate(pred):  # 对每张图片进行检测
            if webcam:  # 如果是摄像头输入
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # 打印字符串
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # 归一化增益
            if det is not None and len(det):
                # 将框从img_size缩放到im0尺寸
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # 打印结果
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # 每类检测数量
                    s += '%g %ss, ' % (n, names[int(c)])  # 添加到字符串

                # 写入结果
                for *xyxy, conf, cls in det:
                    if save_txt:  # 写入文件
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # 归一化xywh
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # 标签格式

                    if save_img or view_img:  # 在图像上添加边界框
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # 打印时间（推理+NMS）
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # 实时显示结果
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # 按q退出
                    raise StopIteration

            # 保存结果（带检测框的图像）
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # 新视频
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # 释放之前的视频写入器

                        fourcc = 'mp4v'  # 输出视频编码
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('结果已保存至 %s' % Path(out))
        if platform.system() == 'Darwin' and not opt.update:  # MacOS
            os.system('open ' + save_path)

    print('完成. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=r'D:\桌面文件\工程部AI推进项目\反光衣检测\best.pt', help='模型权重路径')
    parser.add_argument('--source', type=str, default=r"D:\桌面文件\单人梯子识别.mp4", help='输入源')  # 文件夹或摄像头编号
    parser.add_argument('--output', type=str, default='D:\桌面文件\工程部AI推进项目\反光衣检测\output', help='输出文件夹')  # 输出文件夹
    parser.add_argument('--img-size', type=int, default=640, help='推理图像大小(像素)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='目标置信度阈值')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS的IOU阈值')
    parser.add_argument('--device', default='', help='cuda设备，如0 或 0,1,2,3 或 cpu')
    parser.add_argument('--view-img', action='store_true', help='显示结果')
    parser.add_argument('--save-txt', action='store_true', help='将结果保存为*.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='按类别过滤: --class 0, 或 --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='类无关的NMS')
    parser.add_argument('--augment', action='store_true', help='增强推理')
    parser.add_argument('--update', action='store_true', help='更新所有模型')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # 更新所有模型（修复SourceChangeWarning）
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()