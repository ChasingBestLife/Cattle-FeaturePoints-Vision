from ultralytics import YOLO
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

if __name__ == '__main__':
    # 加载模型
    model = YOLO(r'runs/train/v8_fivebox/weights/best.pt')


    # 评估模型
    results = model.val(data='test.yaml', imgsz=640)
    # 打印评估指标
    # 打印检测框相关指标
    print(f"Box Precision (P): {results.box.p}")
    print(f"Box Recall (R): {results.box.r}")
    print(f"Box mAP50: {results.box.map50}")
    print(f"Box mAP50-95: {results.box.map}")
    # # 打印关键点相关指标
    # print(f"Keypoints OKS: {results.kpts.oks}")
    # print(f"Keypoints mAP50: {results.kpts.map50}")
    # print(f"Keypoints mAP50-95: {results.kpts.map}")