import torch
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import random

random.seed(0)


class DataAugmentationOnDetection:
    def __init__(self):
        super(DataAugmentationOnDetection, self).__init__()

    # def resize_keep_ratio(self, image, boxes, target_size):
    #     # ------------------------------------------------------------------------------------------------------------------
    #     # 未完成
    #     # 参数类型： image：Image.open(path)， boxes:Tensor， target_size:int
    #     # 功能：将图像缩放到size尺寸，调整相应的boxes,同时保持长宽比（最长的边是target size）
    #     # ------------------------------------------------------------------------------------------------------------------
    #     old_size = image.size[0:2]  # 原始图像大小
    #     # 取最小的缩放比例
    #     ratio = min(float(target_size) / (old_size[i]) for i in range(len(old_size)))  # 计算原始图像宽高与目标图像大小的比例，并取其中的较小值
    #     new_size = tuple([int(i * ratio) for i in old_size])  # 根据上边求得的比例计算在保持比例前提下得到的图像大小
    #     # boxes 不用变化，因为是等比例变化
    #     return image.resize(new_size, Image.BILINEAR), boxes


    # def resize(self, img, boxes, size):
    #     # ---------------------------------------------------------
    #     # 未完成
    #     # 类型为 img=Image.open(path)，boxes:Tensor，size:int
    #     # 功能为：将图像长和宽缩放到指定值size，并且相应调整boxes
    #     # ---------------------------------------------------------
    #     w, h = img.size
    #     sw = size / w
    #     sh = size / h
    #     return img.resize((size, size), Image.BILINEAR), boxes * torch.Tensor([sw, sh, sw, sh])
    #
    # def resize_(self, img, boxes, size):
    #     # -----------------------------------------------------------
    #     # 未完成
    #     # 类型为 img=Image.open(path)，boxes:Tensor，size:int
    #     # 功能为：将图像短边缩放到指定值size,保持原有比例不变，并且相应调整boxes
    #     # -----------------------------------------------------------
    #     w, h = img.size
    #     min_size = min(w, h)
    #     sw = sh = size / min_size
    #     ow = int(sw * w + 0.5)
    #     oh = int(sh * h + 0.5)
    #     return img.resize((ow, oh), Image.BILINEAR), boxes * torch.Tensor([sw, sh, sw, sh])


    def random_flip_horizon(self, img, boxes, keypoints, h_rate=0.6):
        # -------------------------------------
        # 随机水平翻转
        # -------------------------------------
        flipped = False
        if np.random.random() < h_rate:
            flipped = True
            transform = transforms.RandomHorizontalFlip(p=1)
            img = transform(img)
            if len(boxes) > 0:
                x = 1 - boxes[:, 1]
                boxes[:, 1] = x
            else:
                # 无标注
                return img, boxes, keypoints, flipped

            # 关键点每 3 个为一组，对每组的第一个值进行水平翻转
            num_keypoints = len(keypoints[0])
            for row in range(len(keypoints)):
                for i in range(0, num_keypoints, 3):
                    if keypoints[row, i] != 0 or keypoints[row, i + 1] != 0:
                        keypoints[row, i] = 1 - keypoints[row, i]
        return img, boxes, keypoints, flipped

    def random_flip_vertical(self, img, boxes, keypoints, v_rate=0.6):
        # -------------------------------------
        # 随机垂直翻转
        # -------------------------------------
        flipped = False
        if np.random.random() < v_rate:
            flipped = True
            transform = transforms.RandomVerticalFlip(p=1)
            img = transform(img)
            if len(boxes) > 0:
                y = 1 - boxes[:, 2]
                boxes[:, 2] = y
            else:
                # 无标注
                return img, boxes, keypoints, flipped

            # 关键点每 3 个为一组，对每组的第一个值进行水平翻转
            num_keypoints = len(keypoints[0])
            for row in range(len(keypoints)):
                for i in range(1, num_keypoints, 3):
                    if keypoints[row, i-1] != 0 or keypoints[row, i] != 0:
                        keypoints[row, i] = 1 - keypoints[row, i]
        return img, boxes, keypoints, flipped

    def random_flip_center(self, img, boxes, keypoints, rate=0.6):
        """
        随机中心旋转（同时进行水平和垂直翻转）
        相当于绕图像中心点旋转180度

        参数:
            img: 输入图像
            boxes: 边界框数据 [N, 5] (class_id, x_center, y_center, width, height)
            keypoints: 关键点数据 [N, K*3] (x, y, visibility)
            rate: 旋转概率
        """
        flipped = False
        if np.random.random() < rate:
            flipped = True

            # 直接实现图像中心旋转
            # 等价于先水平翻转再垂直翻转
            transform = transforms.RandomVerticalFlip(p=1)
            img = transform(img)
            transform = transforms.RandomHorizontalFlip(p=1)
            img = transform(img)

            if len(boxes) > 0:
                # 对边界框的x和y中心坐标进行翻转
                boxes[:, 1] = 1 - boxes[:, 1]  # 水平翻转
                boxes[:, 2] = 1 - boxes[:, 2]  # 垂直翻转
            else:
                # 无标注
                return img, boxes, keypoints, flipped

            # 关键点处理
            # 每3个值为一组 (x, y, visibility)
            num_keypoints = len(keypoints[0])
            for row in range(len(keypoints)):
                for i in range(0, num_keypoints, 3):
                    # 跳过不可见的关键点
                    if keypoints[row, i + 2] == 0:
                        continue
                    # 水平和垂直翻转
                    keypoints[row, i] = 1 - keypoints[row, i]  # x坐标翻转
                    keypoints[row, i + 1] = 1 - keypoints[row, i + 1]  # y坐标翻转

        return img, boxes, keypoints, flipped

    # ------------------------------------------------------
    # 以下img皆为Tensor类型
    # ------------------------------------------------------

    def random_bright(self, img, u=50, p=0.5):
        # -------------------------------------
        # 随机亮度变换
        # -------------------------------------
        flag=False
        if np.random.random() < p:
            flag=True
            alpha = np.random.uniform(-u, u) / 255
            img += alpha
            img = img.clamp(min=0.0, max=1.0)
        return img, flag

    def random_contrast(self, img, lower=0.8, upper=1.2, p=0.7):
        # -------------------------------------
        # 随机增强对比度
        # -------------------------------------
        flag=False
        if np.random.random() < p:
            flag=True
            alpha = np.random.uniform(lower, upper)
            img *= alpha
            img = img.clamp(min=0, max=1.0)
        return img, flag

    def random_saturation(self, img, lower=0.8, upper=1.2, p=0.6):
        # --------------------------------------------------------------------------
        # 随机饱和度变换，针对彩色三通道图像，中间通道乘以一个值
        # --------------------------------------------------------------------------
        flag=False
        if np.random.random() < p:
            flag=True
            alpha = np.random.uniform(lower, upper)
            img[1] = img[1] * alpha
            img[1] = img[1].clamp(min=0, max=1.0)
        return img, flag

    # def center_crop(self, img, boxes, target_size=None):
    #     # -------------------------------------
    #     # 未完成
    #     # 中心裁剪 ，裁剪成 (size, size) 的正方形, 仅限图形，w,h
    #     # 这里用比例是很难算的，转成x1,y1, x2, y2格式来计算
    #     # -------------------------------------
    #     w, h = img.size
    #     size = min(w, h)
    #     if len(boxes) > 0:
    #         # 转换到xyxy格式
    #         label = boxes[:, 0].reshape([-1, 1])
    #         x_, y_, w_, h_ = boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4]
    #         x1 = (w * x_ - 0.5 * w * w_).reshape([-1, 1])
    #         y1 = (h * y_ - 0.5 * h * h_).reshape([-1, 1])
    #         x2 = (w * x_ + 0.5 * w * w_).reshape([-1, 1])
    #         y2 = (h * y_ + 0.5 * h * h_).reshape([-1, 1])
    #         boxes_xyxy = torch.cat([x1, y1, x2, y2], dim=1)
    #         # 边框转换
    #         if w > h:
    #             boxes_xyxy[:, [0, 2]] = boxes_xyxy[:, [0, 2]] - (w - h) / 2
    #         else:
    #             boxes_xyxy[:, [1, 3]] = boxes_xyxy[:, [1, 3]] - (h - w) / 2
    #         in_boundary = [i for i in range(boxes_xyxy.shape[0])]
    #         for i in range(boxes_xyxy.shape[0]):
    #             # 判断x是否超出界限
    #             if (boxes_xyxy[i, 0] < 0 and boxes_xyxy[i, 2] < 0) or (
    #                     boxes_xyxy[i, 0] > size and boxes_xyxy[i, 2] > size):
    #                 in_boundary.remove(i)
    #             # 判断y是否超出界限
    #             elif (boxes_xyxy[i, 1] < 0 and boxes_xyxy[i, 3] < 0) or (
    #                     boxes_xyxy[i, 1] > size and boxes_xyxy[i, 3] > size):
    #                 in_boundary.append(i)
    #         boxes_xyxy = boxes_xyxy[in_boundary]
    #
    #
    #
    #         boxes = boxes_xyxy.clamp(min=0, max=size).reshape([-1, 4])  # 压缩到固定范围
    #         label = label[in_boundary]
    #         # 转换到YOLO格式
    #         x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    #         xc = ((x1 + x2) / (2 * size)).reshape([-1, 1])
    #         yc = ((y1 + y2) / (2 * size)).reshape([-1, 1])
    #         wc = ((x2 - x1) / size).reshape([-1, 1])
    #         hc = ((y2 - y1) / size).reshape([-1, 1])
    #         boxes = torch.cat([xc, yc, wc, hc], dim=1)
    #     # 图像转换
    #     transform = transforms.CenterCrop(size)
    #     img = transform(img)
    #     if target_size:
    #         img = img.resize((target_size, target_size), Image.BILINEAR)
    #     if len(boxes) > 0:
    #         return img, torch.cat([label.reshape([-1, 1]), boxes], dim=1)
    #     else:
    #         return img, boxes

    # 只使用高斯噪声，其它三种噪声，如泊松是与图像亮度成正比，不太适用
    def add_gasuss_noise(self, img, mean=0, std=0.05, p=0.3):
        # -------------------------------------
        # 随机高斯噪声
        # -------------------------------------
        flag=False
        if np.random.random() < p:
            flag=True
            noise = torch.normal(mean, std, img.shape)
            img += noise
            img = img.clamp(min=0, max=1.0)
        return img, flag

    # 粗盐遮挡掩码
    def add_random_mask(self, img, area=0.03, salt_density=0.3, salt_size=1, p=0.3):
        # -------------------------------------
        # 随机位置生成指定面积的粗盐掩码
        # -------------------------------------
        flag = False
        if np.random.random() < p:
            flag = True
            h, w = img.shape[-2:]  # 获取图像的高度和宽度
            max_mask_area = int(h * w * area)  # 计算最大掩码面积（像素数）

            # 随机生成掩码尺寸（确保面积不超过max_mask_area）
            max_side = int(np.sqrt(max_mask_area))
            mask_w = np.random.randint(1, max_side + 1)
            mask_h = min(np.random.randint(1, max_side + 1), max_mask_area // mask_w)

            # 随机选择掩码位置
            x = np.random.randint(0, w - mask_w + 1)
            y = np.random.randint(0, h - mask_h + 1)

            # 创建掩码区域
            mask_area = img[..., y:y + mask_h, x:x + mask_w]

            # 在掩码区域添加粗盐噪声
            salt_mask = torch.rand_like(mask_area) < salt_density
            mask_area[salt_mask] = 1.0  # 将随机选中的像素设为白色（粗盐效果）

            # 确保像素值在有效范围内
            img = img.clamp(min=0, max=1.0)

        return img, flag

    # def add_salt_noise(self, img, p=0.5):
    #     # -------------------------------------
    #     # 随机盐（白点）噪声
    #     # -------------------------------------
    #     flag = False
    #     if np.random.random() < p:
    #         flag = True
    #         noise = torch.rand(img.shape)
    #         alpha = np.random.random() / 5 + 0.7
    #         img[noise[:, :, :] > alpha] = 1.0
    #     return img, flag
    #
    # def add_pepper_noise(self, img, p=0.5):
    #     # -------------------------------------
    #     # 随机椒（黑点）噪声
    #     # -------------------------------------
    #     flag = False
    #     if np.random.random() < p:
    #         flag = True
    #         noise = torch.rand(img.shape)
    #         alpha = np.random.random() / 5 + 0.7
    #         img[noise[:, :, :] > alpha] = 0
    #     return img, flag
    #
    # def add_poisson_noise(self, img, scale_factor=1.0, p=0.5):
    #     # -------------------------------------
    #     # 随机泊松噪声
    #     # -------------------------------------
    #     flag = False
    #     if np.random.random() < p:
    #         flag = True
    #         # 对图像乘以缩放因子以调整强度
    #         scaled_img = img * scale_factor
    #         # 从泊松分布中采样噪声
    #         noisy_img = torch.poisson(scaled_img)
    #         # 再除以缩放因子并将值限制在 [0, 1] 范围内
    #         noisy_img = (noisy_img / scale_factor).clamp(0, 1)
    #
    #     return noisy_img, flag


def plot_pics(img, boxes):
    # 显示图像和候选框，img是Image.Open()类型, boxes是Tensor类型
    plt.imshow(img)
    label_colors = [(213, 110, 89)]
    w, h = img.size
    for i in range(boxes.shape[0]):
        box = boxes[i, 1:]
        xc, yc, wc, hc = box
        x = w * xc - 0.5 * w * wc
        y = h * yc - 0.5 * h * hc
        box_w, box_h = w * wc, h * hc
        plt.gca().add_patch(plt.Rectangle(xy=(x, y), width=box_w, height=box_h,
                                          edgecolor=[c / 255 for c in label_colors[0]],
                                          fill=False, linewidth=2))
    plt.show()


def get_image_list(image_path):
    # 根据图片文件，查找所有图片并返回列表
    files_list = []
    for root, sub_dirs, files in os.walk(image_path):
        for special_file in files:
            special_file = special_file[0: len(special_file)]
            files_list.append(special_file)
    return files_list


def get_label_file(label_path, image_name):
    # 根据图片信息，查找对应的 label
    fname = os.path.join(label_path, image_name[0: len(image_name) - 4] + ".txt")
    data2 = []
    keypoints_data = []
    if not os.path.exists(fname):
        return data2, keypoints_data
    if os.path.getsize(fname) == 0:
        return data2, keypoints_data
    with open(fname, 'r', encoding='utf-8') as infile:
        # 读取并转换标签
        for line in infile:
            data_line = line.strip("\n").split()
            # 提取前 5 个数据
            data2.append([float(i) for i in data_line[:5]])
            # 提取第 6 个及其后面的数据作为关键点数据
            if len(data_line) > 5:
                keypoints = [float(i) for i in data_line[5:]]
                keypoints_data.append(keypoints)
    return data2, keypoints_data


def save_Yolo(img, boxes, keypoints, save_path, prefix, image_name):
    # img: 需要时Image类型的数据， prefix 前缀
    # 将结果保存到save path指示的路径中
    if not os.path.exists(save_path) or \
            not os.path.exists(os.path.join(save_path, "images")):
        os.makedirs(os.path.join(save_path, "images"))
        os.makedirs(os.path.join(save_path, "labels"))
    try:
        img.save(os.path.join(save_path, "images", prefix + image_name))
        with open(os.path.join(save_path, "labels", prefix + image_name[0:len(image_name) - 4] + ".txt"), 'w',
                  encoding="utf-8") as f:
            if len(boxes) > 0:  # 判断是否为空
                # 写入新的label到文件中
                for i, data in enumerate(boxes):
                    str_in = ""
                    for j, a in enumerate(data):
                        if j == 0:
                            str_in += str(int(a))
                        else:
                            str_in += " " + "{:.5f}".format(float(a))
                    # 添加关键点数据
                    if len(keypoints) > i:
                        keypoint_str_list = []
                        for k in range(0, len(keypoints[i]), 3):
                            if int(keypoints[i][k + 2]) == 0:
                                keypoint_str_list.extend([
                                    str(int(keypoints[i][k])),
                                    str(int(keypoints[i][k + 1])),
                                    str(int(keypoints[i][k + 2]))
                                ])
                            else:
                                keypoint_str_list.extend([
                                    "{:.5f}".format(float(keypoints[i][k])),
                                    "{:.5f}".format(float(keypoints[i][k + 1])),
                                    str(int(keypoints[i][k + 2]))
                                ])
                        keypoint_str = " ".join(keypoint_str_list)
                        str_in += " " + keypoint_str
                    f.write(str_in + '\n')
    except:
        print("ERROR: ", image_name, " is bad.")


def runAugumentation(image_path, label_path, save_path):
    image_list = get_image_list(image_path)
    for image_name in image_list:
        print("dealing: " + image_name)
        img = Image.open(os.path.join(image_path, image_name))
        boxes, keypoints = get_label_file(label_path, image_name)
        boxes = torch.tensor(boxes)
        keypoints = torch.tensor(keypoints)

        # 复制数据集到新文件夹
        save_Yolo(img, boxes, keypoints, save_path, prefix="", image_name=image_name)

        # 下面是执行的数据增强功能，可自行选择
        # Image类型的参数
        DAD = DataAugmentationOnDetection()

        """ 尺寸变换   """
        # # 水平旋转
        # t_img, t_boxes, t_keypoints, flag = DAD.random_flip_horizon(img, boxes.clone(), keypoints.clone())
        # if flag != False:
        #     save_Yolo(t_img, t_boxes, t_keypoints, save_path, prefix="fh_", image_name=image_name)
        #     image_name = "fh_"+image_name
        # # 竖直旋转
        # t_img, t_boxes, t_keypoints, flag = DAD.random_flip_vertical(img, boxes.clone(), keypoints.clone())
        # if flag != False:
        #     save_Yolo(t_img, t_boxes, t_keypoints, save_path, prefix="fv_", image_name=image_name)
        #     image_name = "fv_" + image_name

        # 中心旋转
        fc_img, t_boxes, t_keypoints, flag = DAD.random_flip_center(img, boxes.clone(), keypoints.clone())
        if flag != False:
            save_Yolo(fc_img, t_boxes, t_keypoints, save_path, prefix="fc_", image_name=image_name)
            image_name = "fc_"+image_name


        """ 图像变换，用tensor类型"""
        to_tensor = transforms.ToTensor()
        to_image = transforms.ToPILImage()
        # 混合fv+增强
        fc_img = to_tensor(fc_img)

        # fc_img = to_tensor(img)
        # t_boxes = boxes
        # t_keypoints = keypoints

        # 高斯噪声
        gn_t_img, flag = DAD.add_gasuss_noise(fc_img.clone())
        if flag != False:
            save_Yolo(to_image(gn_t_img), t_boxes.clone(), t_keypoints.clone(), save_path, prefix="gn_",
                      image_name=image_name)
            gn_image_name = "gn_" + image_name
        # random_mask 粗盐掩码
        rm_t_img, flag = DAD.add_random_mask(fc_img.clone())
        if flag != False:
            save_Yolo(to_image(rm_t_img), t_boxes.clone(), t_keypoints.clone(), save_path, prefix="rm_",
                      image_name=image_name)
            rm_image_name = "rm_" + image_name

        # random_bright 亮度变化
        rb_img, flag = DAD.random_bright(fc_img.clone())
        if flag != False:
            save_Yolo(to_image(rb_img), t_boxes.clone(), t_keypoints.clone(), save_path, prefix="rb_", image_name=image_name)
            image_name = "rb_" + image_name
        # random_contrast 对比度变化
        rc_img, flag = DAD.random_contrast(rb_img.clone())
        if flag != False:
            save_Yolo(to_image(rc_img), t_boxes.clone(), t_keypoints.clone(), save_path, prefix="rc_", image_name=image_name)
            image_name = "rc_" + image_name
        # random_saturation 饱和度变化
        rs_img, flag = DAD.random_saturation(rc_img.clone())
        if flag != False:
            save_Yolo(to_image(rs_img), t_boxes.clone(), t_keypoints.clone(), save_path, prefix="rs_", image_name=image_name)
            image_name = "rs_" + image_name



        # # add_salt_noise
        # t_img, t_boxes = DAD.add_salt_noise(img.clone()), boxes
        # save_Yolo(to_image(t_img), boxes.clone(), save_path, prefix="sn_", image_name=image_name)
        # # add_pepper_noise
        # t_img, t_boxes = DAD.add_pepper_noise(img.clone()), boxes
        # save_Yolo(to_image(t_img), boxes.clone(), save_path, prefix="pn_", image_name=image_name)

        print("end:     " + image_name)


if __name__ == '__main__':
    # 图像和标签文件夹
    image_path = r"Desktop\1\images"
    label_path = r"Desktop\1\labels"
    save_path = r"Desktop\11"  # 结果保存位置路径，可以是一个不存在的文件夹
    # 运行
    runAugumentation(image_path, label_path, save_path)
