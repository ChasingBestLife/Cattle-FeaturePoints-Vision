import cv2
import numpy as np
import time
from tqdm import tqdm
from ultralytics import YOLO
import matplotlib.pyplot as plt
import torch

# -------------------------------------
# - 预处理
# -------------------------------------
# 载入模型
model = YOLO(r'runs/train/weights/best.pt')
# 载入视频路径
video_path=r'dataset/tt_video.mp4'

# 有 GPU 就用 GPU，没有就用 CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# 切换计算设备
model.to(device)


# 框（rectangle）可视化配置
bbox_color = (150, 0, 0)             # 框的 BGR 颜色
bbox_thickness = 2                   # 框的线宽

# 框类别文字
bbox_labelstr = {
    'font_size':1,         # 字体大小
    'font_thickness':2,   # 字体粗细
    'offset_x':0,          # X 方向，文字偏移距离，向右为正
    'offset_y':-10,       # Y 方向，文字偏移距离，向下为正
}
# 点类别文字
kpt_labelstr = {
    'font_size':0.5,             # 字体大小
    'font_thickness':1,       # 字体粗细
    'offset_x':10,             # X 方向，文字偏移距离，向右为正
    'offset_y':0,            # Y 方向，文字偏移距离，向下为正
}
# 关键点 BGR 配色
kpt_color_map = {
    0:{'name':'left_eye', 'color':[255, 0, 0], 'radius':6},
    1:{'name':'right_eye', 'color':[0, 255, 0], 'radius':6},
    2:{'name':'left_h', 'color':[0, 0, 255], 'radius':6},
    3:{'name':'right_h', 'color':[205, 0, 0], 'radius':6},
    4:{'name':'forehead', 'color':[0, 205, 0], 'radius':6}
}
# 骨架连接 BGR 配色
skeleton_map = [
]

# -------------------------------------
# - 预测视频每一帧图像
# -------------------------------------
def process_frame(img_bgr):
    '''
    输入摄像头画面 bgr-array，输出图像 bgr-array
    '''

    results = model(img_bgr, verbose=False)  # verbose设置为False，不单独打印每一帧预测结果

    # 获取每个框的置信度
    confidences = results[0].boxes.conf
    # print(confidences)

    # 筛选出置信度大于 0.5 的框的索引
    high_conf_indices = torch.where(confidences > 0.5)[0].cpu().numpy().astype('uint32')
    # 转为 numpy array
    # 转成整数的 numpy array
    bboxes_xyxy = results[0].boxes.xyxy.cpu().numpy().astype('uint32')
    bboxes_keypoints = results[0].keypoints.xy.cpu().numpy().astype('uint32')
    # 筛选高置信度
    # 根据 numpy 格式的索引筛选高置信度的边界框和关键点
    if len(high_conf_indices) > 0:
        bboxes_xyxy = bboxes_xyxy[high_conf_indices]
        bboxes_keypoints = bboxes_keypoints[high_conf_indices]

    num_bbox = len(high_conf_indices)
    # print('预测出 {} 个框'.format(num_bbox))

    for idx, idx_label in zip(range(num_bbox), results[0].boxes.cls.cpu().numpy().astype('uint32')):  # 遍历每个框
        # 获取该框坐标
        bbox_xyxy = bboxes_xyxy[idx]
        # 获取框的预测类别（对于关键点检测，只有一个类别）
        bbox_label = results[0].names[idx_label]
        # 画框
        img_bgr = cv2.rectangle(img_bgr, (bbox_xyxy[0], bbox_xyxy[1]), (bbox_xyxy[2], bbox_xyxy[3]), bbox_color,
                                bbox_thickness)
        # 写框类别文字：图片，文字字符串，文字左上角坐标，字体，字体大小，颜色，字体粗细
        img_bgr = cv2.putText(img_bgr, bbox_label,
                              (bbox_xyxy[0] + bbox_labelstr['offset_x'], bbox_xyxy[1] + bbox_labelstr['offset_y']),
                              cv2.FONT_HERSHEY_SIMPLEX, bbox_labelstr['font_size'], bbox_color,
                              bbox_labelstr['font_thickness'])
        bbox_keypoints = bboxes_keypoints[idx]  # 该框所有关键点坐标和置信度

            # 画该框的关键点
        for kpt_id in kpt_color_map:
            # 获取该关键点的颜色、半径、XY坐标
            kpt_color = kpt_color_map[kpt_id]['color']
            kpt_radius = kpt_color_map[kpt_id]['radius']
            kpt_x = bbox_keypoints[kpt_id][0]
            kpt_y = bbox_keypoints[kpt_id][1]

            # 画圆：图片、XY坐标、半径、颜色、线宽（-1为填充）
            img_bgr = cv2.circle(img_bgr, (kpt_x, kpt_y), kpt_radius, kpt_color, -1)

            # 写关键点类别文字：图片，文字字符串，文字左上角坐标，字体，字体大小，颜色，字体粗细
            kpt_label = str(kpt_id)  # 写关键点类别 ID（二选一）
            # kpt_label = str(kpt_color_map[kpt_id]['name']) # 写关键点类别名称（二选一）
            img_bgr = cv2.putText(img_bgr, kpt_label,
                                  (kpt_x + kpt_labelstr['offset_x'], kpt_y + kpt_labelstr['offset_y']),
                                  cv2.FONT_HERSHEY_SIMPLEX, kpt_labelstr['font_size'], kpt_color,
                                  kpt_labelstr['font_thickness'])

    return img_bgr


# -------------------------------------
# - 开始预测图像
# -------------------------------------
def generate_video(input_path):
    filehead = input_path.split('/')[-1]
    output_path = "out-" + filehead

    print('视频开始处理', input_path)

    # 获取视频总帧数
    cap = cv2.VideoCapture(input_path)
    frame_count = 0
    while (cap.isOpened()):
        success, frame = cap.read()
        frame_count += 1
        if not success:
            break
    cap.release()
    print('视频总帧数为', frame_count)

    # cv2.namedWindow('Crack Detection and Measurement Video Processing')
    cap = cv2.VideoCapture(input_path)
    frame_size = (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(output_path, fourcc, fps, (int(frame_size[0]), int(frame_size[1])))

    # 进度条绑定视频总帧数
    with tqdm(total=frame_count - 1) as pbar:
        try:
            while (cap.isOpened()):
                success, frame = cap.read()
                if not success:
                    break

                try:
                    frame = process_frame(frame)
                except:
                    print('error')
                    pass

                if success == True:
                    # cv2.imshow('Video Processing', frame)
                    out.write(frame)

                    # 进度条更新一帧
                    pbar.update(1)

                # if cv2.waitKey(1) & 0xFF == ord('q'):
                # break
        except:
            print('中途中断')
            pass

    cv2.destroyAllWindows()
    out.release()
    cap.release()
    print('视频已保存', output_path)

generate_video(input_path=video_path)