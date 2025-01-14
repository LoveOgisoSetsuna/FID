import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18
import sys
from scipy.linalg import sqrtm
import cv2
import os

from torchvision import models
from torchvision.models import vgg16
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(__file__), "kinetics-i3d"))

from i3d import InceptionI3d


# 1.1 ResNet特征提取器
class FeatureExtractorResNet(nn.Module):
    def __init__(self):
        super(FeatureExtractorResNet, self).__init__()
        base_model = resnet18(pretrained=True)  # 替换为合适的视频模型
        self.model = nn.Sequential(
            *list(base_model.children())[:-1]
        )  # 去掉最后一层全连接

    def forward(self, x):
        with torch.no_grad():
            return self.model(x).squeeze(-1).squeeze(-1)


# 1.2 VGG特征提取器
class FeatureExtractorVGG(nn.Module):
    def __init__(self):
        super(FeatureExtractorVGG, self).__init__()
        base_model = vgg16(pretrained=True)  # 使用预训练的VGG16模型
        self.model = nn.Sequential(*list(base_model.features.children()))  # 提取特征层

    def forward(self, x):
        with torch.no_grad():
            x = self.model(x)  # 前向传播
            x = F.adaptive_avg_pool2d(x, (1, 1))  # 使用全局池化层
            return x.view(x.size(0), -1)  # 拉平为一维向量


# 1.3 I3D特征提取器
class FeatureExtractorI3D(nn.Module):
    def __init__(self, i3d_model_path=None):
        super(FeatureExtractorI3D, self).__init__()
        self.model = InceptionI3d(
            400, spatial_squeeze=True
        )  # 假设分类为400类，可以调整

        if i3d_model_path:
            # 加载预训练模型
            self.model.load(torch.load(i3d_model_path))
            self.model.eval()

    def forward(self, x):
        with torch.no_grad():
            return self.model.extract_features(x)


# 1.4 InceptionV3特征提取器
class FeatureExtractorInceptionV3(nn.Module):
    def __init__(self):
        super(FeatureExtractorInceptionV3, self).__init__()
        self.model = models.inception_v3(
            pretrained=True, transform_input=False
        )  # 加载InceptionV3模型
        self.model.eval()

    def forward(self, x):
        with torch.no_grad():
            x = self.model.Conv2d_1a_3x3(x)
            x = self.model.Conv2d_2a_3x3(x)
            x = self.model.Conv2d_2b_3x3(x)
            x = self.model.maxpool1(x)
            x = self.model.Conv2d_3b_1x1(x)
            x = self.model.Conv2d_4a_3x3(x)
            x = self.model.maxpool1(x)
            x = self.model.Mixed_5b(x)
            x = self.model.Mixed_5c(x)
            x = self.model.Mixed_5d(x)
            x = self.model.Mixed_6a(x)
            x = self.model.Mixed_6b(x)
            x = self.model.Mixed_6c(x)
            x = self.model.Mixed_6d(x)
            x = self.model.Mixed_6e(x)
            x = self.model.Mixed_7a(x)
            x = self.model.Mixed_7b(x)
            x = self.model.Mixed_7c(x)
            x = self.model.avgpool(x)
            return x.view(x.size(0), -1)


# 1.5 BigGan特征提取器
class FeatureExtractorBigGAN(nn.Module):
    def __init__(self):
        super(FeatureExtractorBigGAN, self).__init__()
        # 使用torch hub加载BigGAN模型
        self.model = load(
            "facebookresearch/BigGAN-PyTorch", "biggan-deep-256"
        )  # BigGAN模型

    def forward(self, x):
        with torch.no_grad():
            # BigGAN模型是基于Tensor的，转成合适格式进行特征提取
            return self.model.forward(x)[0].flatten(1)  # 获取最后一层flatten特征


# 2. 提取特征
def extract_features(video_paths, model, device, frame_size=(112, 112)):
    features = []
    for path in video_paths:
        video_path = os.path.join(os.path.dirname(__file__), "input", path)
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print("Error: Could not open video file.")

        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, frame_size)  # 调整尺寸
            frame = frame[:, :, ::-1]  # BGR to RGB
            frames.append(frame)

        cap.release()

        # 转换为张量并归一化
        frames = np.array(frames) / 255.0
        frames = (
            torch.tensor(frames, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
        )
        feature = model(frames).mean(dim=0)  # 对帧的特征取平均值
        features.append(feature.cpu().numpy())

    return np.array(features)


# 3. 计算均值和协方差矩阵
def calculate_statistics(features):
    # 确保features是二维的，样本为行，特征为列
    if features.ndim == 1:
        features = features.reshape(1, -1)  # 如果是1D数组，转为1个样本的二维数据
    elif features.ndim > 2:
        raise ValueError("Input features must be 1D or 2D")

    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)

    # 输出协方差矩阵的形状，便于调试
    print(f"Covariance matrix shape: {sigma.shape}")

    return mu, sigma


# 4. 计算FID
# 计算FID
def calculate_fid(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2

    # 确保协方差矩阵是二维的
    if sigma1.ndim == 1:
        sigma1 = np.expand_dims(sigma1, axis=0)
    if sigma2.ndim == 1:
        sigma2 = np.expand_dims(sigma2, axis=0)

    # 确保协方差矩阵是二维矩阵
    if sigma1.ndim != 2 or sigma2.ndim != 2:
        raise ValueError("Covariance matrices must be 2-dimensional.")

    try:
        # 计算协方差矩阵的平方根
        covmean = sqrtm(sigma1.dot(sigma2))
    except ValueError:
        # 如果发生错误，尝试使用伪逆（用于奇异矩阵）
        print("Warning: Covariance matrix is singular, using pseudo-inverse.")
        covmean = sqrtm(np.linalg.pinv(sigma1).dot(np.linalg.pinv(sigma2)))

    # 如果结果是复数，则取其实部
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # 计算FID
    fid = np.sum(diff**2) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid


# 主函数
if __name__ == "__main__":

    # 选择使用的模型类型
    model_type = "ResNet"
    # 选择使用的视频组
    video_group = 2

    if video_group == 1:
        # 视频路径列表(原视频/生成视频组)
        real_videos = [
            "fsgq_real_video.mp4",
            "fsgq_real_video.mp4",
            "fsgq_real_video.mp4",
            "fsgq_real_video.mp4",
            "fsgq_real_video.mp4",
        ]
        generated_videos = [
            "mimic1.mp4",
            "mimic2.mp4",
            "mimic3.mp4",
            "mimic4.mp4",
            "mimic5.mp4",
        ]
    elif video_group == 2:
        # 视频路径列表(原视频/自生成视频组)
        real_videos = [
            "fsgq_real_video.mp4",
            "fsgq_real_video.mp4",
            "fsgq_real_video.mp4",
            "fsgq_real_video.mp4",
            "fsgq_real_video.mp4",
        ]
        generated_videos = [
            "fsgq_self_generate.mp4",
            "fsgq_self_generate.mp4",
            "fsgq_self_generate.mp4",
            "fsgq_self_generate.mp4",
            "fsgq_self_generate.mp4",
        ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_type == "ResNet":
        model = FeatureExtractorResNet().to(device)
    elif model_type == "VGG16":
        model = FeatureExtractorVGG().to(device)
    elif model_type == "I3D":
        i3d_model_path = os.path.join(
            os.path.dirname(__file__), "model-i3d", "model_rgb.pth"
        )
        model = FeatureExtractorI3D(i3d_model_path).to(device)
    elif model_type == "InceptionV3":
        model = FeatureExtractorInceptionV3().to(device)
    else:
        raise ValueError(
            "Invalid model type. Choose 'ResNet' , 'VGG16' ,  'I3D'  or  'InceptionV3' ."
        )

    # 提取真实视频和生成视频的特征
    real_features = extract_features(real_videos, model, device)
    generated_features = extract_features(generated_videos, model, device)

    # 计算统计量
    mu_real, sigma_real = calculate_statistics(real_features)
    mu_generated, sigma_generated = calculate_statistics(generated_features)

    # 计算FID
    fid_score = calculate_fid(mu_real, sigma_real, mu_generated, sigma_generated)
    # if fid_score < 1:
    #     print(f"FID-R Score: {1-fid_score}")
    # elif fid_score > 1:
    print(f"FID Score: {fid_score}")
