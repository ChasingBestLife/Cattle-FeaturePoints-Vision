from ultralytics import YOLO
import torch
import cv2
import matplotlib.pyplot as plt
import os


# -------------------------------------
# - 预处理
# -------------------------------------

# 载入模型
model = YOLO(r"best.pt")
# 定义测试图片文件夹路径
test_folder = r"测试集"
# 定义输出文件夹路径
output_folder = os.path.join('test', 'output', 'img')
# 创建输出文件夹
os.makedirs(output_folder, exist_ok=True)
# 有 GPU 就用 GPU，没有就用 CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# 切换计算设备
model.to(device)
# print(model.names)


# -------------------------------------
# - 使用官方的predict方法，预测图像
# -------------------------------------

# 对指定的图像文件夹进行推理，并设置各种参数
results = model.predict(
    source=test_folder,  # 数据来源，可以是文件夹、图片路径、视频、URL，或设备ID（如摄像头）
    conf=0.6,  # 置信度阈值
    iou=0.6,  # IoU 阈值
    half=False,  # 使用半精度推理
    device='0',  # 使用设备，None 表示自动选择，比如'cpu','0'
    max_det=500,  # 最大检测数量
    vid_stride=1,  # 视频帧跳跃设置
    stream_buffer=False,  # 视频流缓冲
    visualize=False,  # 可视化模型特征
    augment=False,  # 启用推理时增强
    agnostic_nms=False,  # 启用类无关的NMS
    classes=None,  # 指定要检测的类别
    retina_masks=False,  # 使用高分辨率分割掩码
    embed=None,  # 提取特征向量层
    show=True,  # 是否显示推理图像
    save=True,  # 保存推理结果
    save_frames=False,  # 保存视频的帧作为图像
    save_txt=True,  # 保存检测结果到文本文件
    save_conf=False,  # 保存置信度到文本文件
    save_crop=False,  # 保存裁剪的检测对象图像
    show_labels=True,  # 显示检测的标签
    show_conf=False,  # 显示检测置信度
    show_boxes=True,  # 显示检测框
    line_width=1,  # 设置边界框的线条宽度，比如2，4
    name='v8_fivebox_result'
)


# -------------------------------------
# - 自定义方法，预测图像
# -------------------------------------
#
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# # print('device:', device)
#
# # 载入模型
# # model = YOLO('best.pt')
# # 切换计算设备
# model.to(device)
# # print(model.device)
#
# # 预测
# # 传入图像、视频、摄像头ID（对应命令行的 source 参数）
# # img_path = r"Desktop\R-C.jpg"
# img_path = r'Desktop\1\frame_001213.jpg'
# results = model(img_path)
# # 解析预测结果
# print(results[0])
# # 解析目标检测预测结果
# # 预测框的所有类别（MS COCO数据集八十类）
# print(results[0].names)
# # 预测类别 ID
# print(results[0].boxes.cls)
#
# num_bbox = len(results[0].boxes.cls)
# print('预测出 {} 个框'.format(num_bbox))
# # 每个框的置信度
# print(results[0].boxes.conf)
#
# # 转为 numpy array
# # 转成整数的 numpy array
# bboxes_xyxy = results[0].boxes.xyxy.cpu().numpy().astype('uint32')
# bboxes_keypoints = results[0].keypoints.xy.cpu().numpy().astype('uint32')
#
# # OpenCV可视化关键点
# img_bgr = cv2.imread(img_path)
# plt.imshow(img_bgr[:,:,::-1])
# plt.savefig('output.png')  # 保存图像到文件
# # plt.show()  # 注释掉显示图像的代码
#
# # 框（rectangle）可视化配置
# bbox_color = (150, 0, 0)             # 框的 BGR 颜色
# bbox_thickness = 1                   # 框的线宽
#
# # 框类别文字
# bbox_labelstr = {
#     'font_size':1,         # 字体大小
#     'font_thickness':1,   # 字体粗细
#     'offset_x':0,          # X 方向，文字偏移距离，向右为正
#     'offset_y':-80,        # Y 方向，文字偏移距离，向下为正
# }
# # 关键点 BGR 配色
# kpt_color_map = {
#     0:{'name':'left_eye', 'color':[255, 0, 0], 'radius':6},
#     1:{'name':'right_eye', 'color':[0, 255, 0], 'radius':6},
#     2:{'name':'left_h', 'color':[0, 0, 255], 'radius':6},
#     3:{'name':'right_h', 'color':[205, 0, 0], 'radius':6},
#     4:{'name':'forehead', 'color':[0, 205, 0], 'radius':6}
# }
#
#
# # 点类别文字
# kpt_labelstr = {
#     'font_size':0,             # 字体大小
#     'font_thickness':0,       # 字体粗细
#     'offset_x':30,             # X 方向，文字偏移距离，向右为正
#     'offset_y':120,            # Y 方向，文字偏移距离，向下为正
# }
#
# # 骨架连接 BGR 配色
# skeleton_map = [
# ]
#
# for idx in range(num_bbox):  # 遍历每个框
#     # 获取该框坐标
#     bbox_xyxy = bboxes_xyxy[idx]
#     # 获取框的预测类别（对于关键点检测，只有一个类别）
#     bbox_label = results[0].names[idx]
#     # 画框
#     img_bgr = cv2.rectangle(img_bgr, (bbox_xyxy[0], bbox_xyxy[1]), (bbox_xyxy[2], bbox_xyxy[3]), bbox_color,
#                             bbox_thickness)
#
#     # 写框类别文字：图片，文字字符串，文字左上角坐标，字体，字体大小，颜色，字体粗细
#     img_bgr = cv2.putText(img_bgr, bbox_label,
#                           (bbox_xyxy[0] + bbox_labelstr['offset_x'], bbox_xyxy[1] + bbox_labelstr['offset_y']),
#                           cv2.FONT_HERSHEY_SIMPLEX, bbox_labelstr['font_size'], bbox_color,
#                           bbox_labelstr['font_thickness'])
#
#     bbox_keypoints = bboxes_keypoints[idx]  # 该框所有关键点坐标和置信度
#
#     # 画该框的关键点
#     for kpt_id in kpt_color_map:
#         # 获取该关键点的颜色、半径、XY坐标
#         kpt_color = kpt_color_map[kpt_id]['color']
#         kpt_radius = kpt_color_map[kpt_id]['radius']
#         kpt_x = bbox_keypoints[kpt_id][0]
#         kpt_y = bbox_keypoints[kpt_id][1]
#
#         # 画圆：图片、XY坐标、半径、颜色、线宽（-1为填充）
#         img_bgr = cv2.circle(img_bgr, (kpt_x, kpt_y), kpt_radius, kpt_color, -1)
#
#         # 写关键点类别文字：图片，文字字符串，文字左上角坐标，字体，字体大小，颜色，字体粗细
#         # kpt_label = str(kpt_id) # 写关键点类别 ID
#         kpt_label = str(kpt_color_map[kpt_id]['name'])  # 写关键点类别名称
#         img_bgr = cv2.putText(img_bgr, kpt_label, (kpt_x + kpt_labelstr['offset_x'], kpt_y + kpt_labelstr['offset_y']),
#                               cv2.FONT_HERSHEY_SIMPLEX, kpt_labelstr['font_size'], kpt_color,
#                               kpt_labelstr['font_thickness'])
# plt.imshow(img_bgr[:,:,::-1])
# plt.savefig('output1.png')  # 保存图像到文件
# # plt.show()  # 注释掉显示图像的代码
# cv2.imwrite('C1_output.jpg', img_bgr)
#
# i=1
# # 遍历文件夹中的所有图片
# for filename in os.listdir(test_folder):
#     if filename.endswith(('.jpg', '.jpeg', '.png')):
#         img_path = os.path.join(test_folder, filename)
#
#         # 预测
#         results = model(img_path)
#         # 解析预测结果
#         print(results[0])
#         # 解析目标检测预测结果
#         # 预测框的所有类别
#         print(results[0].names)
#         # 预测类别 ID
#         print(results[0].boxes.cls)
#
#         # num_bbox = len(results[0].boxes.cls)
#         # print('预测出 {} 个框'.format(num_bbox))
#         # 获取每个框的置信度
#         confidences = results[0].boxes.conf
#         print(confidences)
#
#         # 筛选出置信度大于 0.6 的框的索引
#         high_conf_indices = torch.where(confidences > 0.6)[0].cpu().numpy().astype('uint32')
#         # 转为 numpy array
#         # 转成整数的 numpy array
#         bboxes_xyxy = results[0].boxes.xyxy.cpu().numpy().astype('uint32')
#         bboxes_keypoints = results[0].keypoints.xy.cpu().numpy().astype('uint32')
#         # 筛选高置信度
#         # 根据 numpy 格式的索引筛选高置信度的边界框和关键点
#         if len(high_conf_indices) > 0:
#             bboxes_xyxy = bboxes_xyxy[high_conf_indices]
#             bboxes_keypoints = bboxes_keypoints[high_conf_indices]
#
#         num_bbox = len(high_conf_indices)
#         print('预测出 {} 个框'.format(num_bbox))
#
#         #     print(f"筛选后有 {len(high_conf_indices)} 个高置信度的框")
#         #     print("高置信度边界框坐标：")
#         #     print(bboxes_xyxy)
#         #     print("高置信度关键点坐标：")
#         #     print(bboxes_keypoints)
#         # else:
#         #     print("没有置信度高于 0.6 的框")
#
#         # OpenCV可视化关键点
#         img_bgr = cv2.imread(img_path)
#         plt.imshow(img_bgr[:, :, ::-1])
#         output_filename = os.path.splitext(filename)[0] + '_output.png'
#         output_file_path = os.path.join(output_folder, output_filename)
#         # plt.savefig(output_file_path)  # 保存图像到文件
#         # plt.show()  # 注释掉显示图像的代码
#
#         # 框（rectangle）可视化配置
#         bbox_color = (150, 0, 0)  # 框的 BGR 颜色
#         bbox_thickness = 1  # 框的线宽
#
#         # 框类别文字
#         bbox_labelstr = {
#             'font_size': 1,  # 字体大小
#             'font_thickness': 1,  # 字体粗细
#             'offset_x': 0,  # X 方向，文字偏移距离，向右为正
#             'offset_y': -80,  # Y 方向，文字偏移距离，向下为正
#         }
#         # 关键点 BGR 配色
#         kpt_color_map = {
#             0: {'name': 'left_eye', 'color': [255, 0, 0], 'radius': 6},
#             1: {'name': 'right_eye', 'color': [0, 255, 0], 'radius': 6},
#             2: {'name': 'left_h', 'color': [0, 0, 255], 'radius': 6},
#             3: {'name': 'right_h', 'color': [205, 0, 0], 'radius': 6},
#             4: {'name': 'forehead', 'color': [0, 205, 0], 'radius': 6}
#         }
#
#         # 点类别文字
#         kpt_labelstr = {
#             'font_size': 0,  # 字体大小
#             'font_thickness': 0,  # 字体粗细
#             'offset_x': 30,  # X 方向，文字偏移距离，向右为正
#             'offset_y': 120,  # Y 方向，文字偏移距离，向下为正
#         }
#
#         # 骨架连接 BGR 配色
#         skeleton_map = [
#             # {'srt_kpt_id':18, 'dst_kpt_id':3, 'color':[196, 75, 255], 'thickness':5},
#             # {'srt_kpt_id':18, 'dst_kpt_id':4, 'color':[196, 75, 255], 'thickness':5},
#         ]
#
#         for idx, idx_label in zip(range(num_bbox), results[0].boxes.cls.cpu().numpy().astype('uint32')):  # 遍历每个框
#             # 获取该框坐标
#             bbox_xyxy = bboxes_xyxy[idx]
#             # 获取框的预测类别（对于关键点检测，只有一个类别）
#             bbox_label = results[0].names[idx_label]
#             # 画框
#             img_bgr = cv2.rectangle(img_bgr, (bbox_xyxy[0], bbox_xyxy[1]), (bbox_xyxy[2], bbox_xyxy[3]), bbox_color,
#                                     bbox_thickness)
#
#             # 写框类别文字：图片，文字字符串，文字左上角坐标，字体，字体大小，颜色，字体粗细
#             img_bgr = cv2.putText(img_bgr, bbox_label,
#                                   (bbox_xyxy[0] + bbox_labelstr['offset_x'], bbox_xyxy[1] + bbox_labelstr['offset_y']),
#                                   cv2.FONT_HERSHEY_SIMPLEX, bbox_labelstr['font_size'], bbox_color,
#                                   bbox_labelstr['font_thickness'])
#
#             bbox_keypoints = bboxes_keypoints[idx]  # 该框所有关键点坐标和置信度
#
#             # 画该框的关键点
#             for kpt_id in kpt_color_map:
#                 # 获取该关键点的颜色、半径、XY坐标
#                 kpt_color = kpt_color_map[kpt_id]['color']
#                 kpt_radius = kpt_color_map[kpt_id]['radius']
#                 kpt_x = bbox_keypoints[kpt_id][0]
#                 kpt_y = bbox_keypoints[kpt_id][1]
#
#                 # 画圆：图片、XY坐标、半径、颜色、线宽（-1为填充）
#                 img_bgr = cv2.circle(img_bgr, (kpt_x, kpt_y), kpt_radius, kpt_color, -1)
#
#                 # 写关键点类别文字：图片，文字字符串，文字左上角坐标，字体，字体大小，颜色，字体粗细
#                 # kpt_label = str(kpt_id) # 写关键点类别 ID
#                 kpt_label = str(kpt_color_map[kpt_id]['name'])  # 写关键点类别名称
#                 img_bgr = cv2.putText(img_bgr, kpt_label,
#                                       (kpt_x + kpt_labelstr['offset_x'], kpt_y + kpt_labelstr['offset_y']),
#                                       cv2.FONT_HERSHEY_SIMPLEX, kpt_labelstr['font_size'], kpt_color,
#                                       kpt_labelstr['font_thickness'])
#         plt.imshow(img_bgr[:, :, ::-1])
#         # output_filename = os.path.splitext(filename)[0] + '_output1.png'
#         output_filename = f'ouput_{i}.png'
#         i+=1
#         output_file_path = os.path.join(output_folder, output_filename)
#         plt.savefig(output_file_path)  # 保存图像到文件
#         # plt.show()  # 注释掉显示图像的代码
#         # cv2_output_filename = os.path.splitext(filename)[0] + '_C1_output.jpg'
#         # cv2_output_file_path = os.path.join(output_folder, cv2_output_filename)
#         # cv2.imwrite(cv2_output_file_path, img_bgr)
