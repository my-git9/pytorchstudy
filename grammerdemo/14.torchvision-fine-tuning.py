import torch
import torchvision.models as models

# 加载预训练模型
googlenet = models.googlenet(pretrained=True)

# 提取分类层的输入参数
fc_in_features = googlenet.fc.in_features
print("fc_in_features: ", fc_in_features)

# 查看分类层的输出参数
fc_out_features = googlenet.fc.out_features
print("fc_out_features: ", fc_out_features)

# 修改预训练模型的输出分类数
googlenet.fc = torch.nn.Linear(fc_in_features, 10)

