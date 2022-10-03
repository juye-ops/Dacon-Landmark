import torch
import torch.nn as nn
import torchvision.models as models

from torchvision import transforms


class CustomModel(nn.Module):
    def __init__(self, device, num_classes):
        super(CustomModel, self).__init__()
        # Image

#        self.augment = nn.Sequential(
#            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#            transforms.RandomApply([
#                transforms.RandomRotation((-30, 30)),
#                transforms.RandomPerspective(distortion_scale=0.3, p=0.8),
#            ]),
#            transforms.RandomHorizontalFlip(0.3),
#            transforms.RandomInvert(0.2),
#        )
        self.cnn_extract = nn.Sequential(
 #           self.augment,
            models.vit_b_16(pretrained=True).to(device),
        )

        self.classifier = nn.Sequential(
            nn.Linear(1000, num_classes)
            # 선형회귀. 4160개의 입력으로 num_classes, 즉 cat3의 종류 개수만큼의 출력
            # 근데 왜 4160개? "4160 - 1024 = 3136"이고 "3136 / 64 = 49". 즉 이미지는 "7*7*64"로 출력됨.
        )

    def forward(self, img):
        img_feature = self.cnn_extract(img)  # cnn_extract 적용
        img_feature = torch.flatten(img_feature, start_dim=1)  # 1차원으로 변환
        # text_feature = self.nlp_extract(text)  # nlp_extract 적용
        # feature = torch.cat([img_feature, text_feature], axis=1)  # 2개 연결(3136 + 1024)
        # output = self.classifier(feature)  # classifier 적용
        # 2개 연결(3136 + 1024)
        output = self.classifier(img_feature)  # classifier 적용
        return output

# from coca_pytorch.coca_pytorch import CoCa
# from vit_pytorch import ViT
# from vit_pytorch.extractor import Extractor
#
#
# def coca(num_classes):
#     vit = ViT(
#         image_size=256,
#         patch_size=32,
#         num_classes=num_classes,
#         dim=1024,
#         depth=6,
#         heads=16,
#         mlp_dim=2048)
#     vit = Extractor(vit, return_embeddings_only=True, detach=False)
#
#     return CoCa(
#         dim=512,  # model dimension
#         img_encoder=vit,  # vision transformer - image encoder, returning image embeddings as (batch, seq, dim)
#         image_dim=1024,  # image embedding dimension, if not the same as model dimensions
#         num_tokens=20000,  # number of text tokens
#         unimodal_depth=6,  # depth of the unimodal transformer
#         multimodal_depth=6,  # depth of the multimodal transformer
#         dim_head=64,  # dimension per attention head
#         heads=8,  # number of attention heads
#         caption_loss_weight=1.,  # weight on the autoregressive caption loss
#         contrastive_loss_weight=1.,  # weight on the contrastive loss between image and text CLS embeddings
#     )