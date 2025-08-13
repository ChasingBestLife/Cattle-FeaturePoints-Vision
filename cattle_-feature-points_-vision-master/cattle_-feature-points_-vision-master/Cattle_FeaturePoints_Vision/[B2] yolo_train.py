import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # model = YOLO("custom.pt")  # load a pretrained model (recommended for training)
    model = YOLO("ultralytics/yolov8s-mypose-CA.yaml")

    model.train(data=r"./data.yaml",
                # 如果大家任务是其它的'ultralytics/cfg/default.yaml'找到这里修改task可以改成detect, segment, classify, pose
                cache=False,
                imgsz=640,
                epochs=500,
                batch=12,
                close_mosaic=0,
                workers=4,
                device='0',
                optimizer='SGD',  # using SGD 优化器 默认为auto建议大家  使用固定的.
                # resume=, # 续训的话这里填写True, yaml文件的地方改为lats.pt的地址,需要注意的是如果你设置训练200轮次模型训练了200轮次是没有办法进行续训的.
                amp=True,  # 如果出现训练损失为Nan可以关闭amp
                project='runs/train',
                freeze=0,
                name='v8_fivebox'
                )