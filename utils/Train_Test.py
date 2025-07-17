from tqdm import tqdm
import torch

def train_one_epoch(dataloader, model, loss_fn, optimizer, device):
    model.train()
    total_loss, correct = 0.0, 0
    total = len(dataloader.dataset)
    loader = tqdm(dataloader, desc='Train', unit='batch')
    for X, y in loader:
        X, y = X.to(device).float(), y.to(device)
        optimizer.zero_grad()
        outputs = model(X)
        loss = loss_fn(outputs, y)
        loss.backward()
        optimizer.step()
        loader.set_postfix(loss=f'{loss.item():.4f}')
        total_loss += loss.item() * X.size(0)
        correct += (outputs.argmax(1) == y).sum().item()
    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc

def evaluate(dataloader, model, loss_fn, device):
    model.eval()
    total_loss, correct = 0.0, 0
    total = len(dataloader.dataset)
    loader = tqdm(dataloader, desc='Test ', unit='batch')
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device).float(), y.to(device)
            outputs = model(X)
            total_loss += loss_fn(outputs, y).item() * X.size(0)
            correct += (outputs.argmax(1) == y).sum().item()
    avg_loss = total_loss / total
    acc = correct / total
    print(f"Test Error: Accuracy: {100 * acc:>0.1f}%, Avg loss: {avg_loss:.4f}\n")
    return avg_loss, acc