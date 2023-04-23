import torch.nn as nn
from torchvision import models

class ImageEncoder(nn.Module):
    def __init__(self, embedding_size):
        super(ImageEncoder, self).__init__()
        self.vgg16 = models.vgg16(pretrained=True)
        self.resnet50 = models.resnet50(pretrained=True)
        self.vgg16.classifier = nn.Sequential(*list(self.vgg16.classifier.children())[:-1])
        self.fc = nn.Linear(self.resnet50.fc.in_features, embedding_size)
    
    def forward(self, x):
        x = self.vgg16(x)
        x = self.resnet50(x)
        x = self.fc(x)
        return x

class CaptionEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers):
        super(CaptionEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, batch_first=True)
    
    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = x[:, -1, :] # 句の最後の隠れ状態を使用する
        return x
