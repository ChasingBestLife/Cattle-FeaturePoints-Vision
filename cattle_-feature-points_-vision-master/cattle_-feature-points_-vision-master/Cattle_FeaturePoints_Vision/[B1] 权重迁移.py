"""

*****************************************************
* 迁移官方预训练权重
*****************************************************

"""
import torch

yolov8 = torch.load('yolov8s-pose.pt', weights_only=False)
best_pt = torch.load('best.pt', weights_only=False)

yolov8_dict = yolov8['model'].state_dict()
best_pt_dict = best_pt['model'].state_dict()
# print(len(yolov8_dict), len(best_pt_dict))

# print(list(yolov8_dict.keys())[:10])
from copy import deepcopy
new_dict = {}
for key, value in yolov8_dict.items():
    key_list = key.split('.')
    if int(key_list[1]) < 7:
        new_dict[key] = deepcopy(value)
        continue
    # elif int(key_list[1]) > 12 and int(key_list[1]) < 16:
    #     key_list[1] = str(int(key_list[1]) + 5)
    #     new_dict[".".join(key_list)] = deepcopy(value)
    #     continue
    else:
        continue

best_pt['model'].load_state_dict(new_dict, strict=False)

# # 1. 解冻特定层（例如：model.34.dfl.conv.weight）
# for name, param in best_pt['model'].named_parameters():
#     if "model.34.dfl.conv.weight" in name:
#         param.requires_grad = True  # 解冻该层，允许梯度更新
#
# for name, param in best_pt['model'].named_parameters():
#     if "model.34.dfl.conv.weight" in name:
#         print(param.requires_grad)
# # 2. 验证解冻结果
# for name, param in best_pt['model'].named_parameters():
#     print(f"参数: {name}, 是否可训练: {param.requires_grad}")


torch.save(best_pt, 'custom.pt')