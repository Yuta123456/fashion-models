import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, outputs, labels):
        # outputs は2つの埋め込みベクトルを含むタプルです
        # labels は、2つの入力が同じクラスかどうかを示す0または1のテンソルです
        # outputs[0] が画像の埋め込みベクトル、outputs[1] がキャプションの埋め込みベクトルです
        image_embeddings, caption_embeddings = outputs[0], outputs[1]
        
        # コサイン類似度を計算します
        # 2つの埋め込みベクトルがどの程度類似しているかを表します
        sim = F.cosine_similarity(image_embeddings, caption_embeddings)
        
        # lossを計算します
        # 同じクラスの場合は距離が小さくなり、違うクラスの場合は距離が大きくなるようにします
        loss = torch.mean((1 - labels) * torch.pow(sim, 2) +
                          (labels) * torch.pow(torch.clamp(self.margin - sim, min=0.0), 2))
        return loss