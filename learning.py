import torch
from torch import optim
from torchvision import transforms
from torch.utils.data import DataLoader

from ContrastiveLoss import ContrastiveLoss

from EmbeddingDataset import EmbeddingDataset
from JointEmbedding import CaptionEncoder, ImageEncoder

# 乱数のシードを設定する
torch.manual_seed(0)

# ハイパーパラメータを設定する
num_epochs = 10
batch_size = 32
learning_rate = 0.001
embedding_size = 1000
hidden_size = 3
vocab_size = 30000
num_layers = 64
# データセットの読み込み
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# targetTransformに、lstm自体のモデルを入れるべきなのか？
train_dataset = EmbeddingDataset("annotation_file", "img_dir", transform=transform, target_transform=None)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# モデルの定義
image_model = ImageEncoder(embedding_size=embedding_size)
# TODO:
text_model = CaptionEncoder(vocab_size, embedding_size, hidden_size, num_layers)

# 損失関数と最適化アルゴリズムの定義
criterion = ContrastiveLoss()
optimizer_image = optim.Adam(image_model.parameters(), lr=learning_rate)
optimizer_text = optim.Adam(text_model.parameters(), lr=learning_rate)

# 学習ループ
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        inputs, labels = data
        optimizer_image.zero_grad()
        optimizer_text.zero_grad()
        image_embedding = image_model(inputs)
        text_embedding = image_model(inputs)

        # TODO
        loss = criterion((image_embedding, text_embedding), 1)
        loss.backward()
        optimizer_image.step()
        optimizer_text.step()
        running_loss += loss.item()
    epoch_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

print('Finished Training')
