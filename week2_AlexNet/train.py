import torch
import torch.nn as nn
import torch.optim as optim
from models.alexnet import AlexNet
from utils.dataloader import get_loaders
from utils.metrics import compute_topk_accuracy

def train():
    print("🚀 [Start] Training begins...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")

    model = AlexNet(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.9,
        weight_decay=0.0005
    )

    lr_schedule = {30: 0.001, 60: 0.0001}

    print("📦 Loading data...")
    train_loader, test_loader = get_loaders(batch_size=128, use_subset=True)
    print("✅ Data loaded!")

    for epoch in range(1, 11):
        print(f"\n📚 [Epoch {epoch}/10] ------------------------------")
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        # 학습률 조정
        if epoch in lr_schedule:
            for g in optimizer.param_groups:
                g['lr'] = lr_schedule[epoch]
            print(f"🔧 Updated learning rate: {lr_schedule[epoch]}")

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # 미니배치 진행 로그 (10% 간격)
            if batch_idx % (len(train_loader) // 10 + 1) == 0:
                print(f"  🔄 Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

        acc = 100. * correct / total
        print(f"📊 Epoch {epoch} Summary: Loss = {total_loss:.3f}, Accuracy = {acc:.2f}%")

        if epoch % 10 == 0:
            test(model, test_loader, device)

    print("\n🎉 [Done] Training complete!")

    from utils.visualize import (
        visualize_first_conv_weights,
        get_feature_vectors,
        show_nearest_neighbors
    )

    print("🖼️ Visualizing Conv1 filters...")
    visualize_first_conv_weights(model)

    print("🔍 Computing nearest neighbors...")
    features, labels, images = get_feature_vectors(model, test_loader, device)
    show_nearest_neighbors(query_idx=42, features=features, images=images, labels=labels, k=6)


def test(model, loader, device):
    print("🧪 Running evaluation on test set...")
    model.eval()
    top1_total = 0
    top5_total = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            top1, top5 = compute_topk_accuracy(outputs, labels, topk=(1, 5))
            batch_size = labels.size(0)
            top1_total += top1 * batch_size
            top5_total += top5 * batch_size
            total += batch_size

    print(f"✅ [Test Results] Top-1 Accuracy: {top1_total / total:.2f}%, Top-5 Accuracy: {top5_total / total:.2f}%")


if __name__ == '__main__':
    train()
