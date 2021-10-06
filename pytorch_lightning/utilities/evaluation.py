import torch


def get_loss(model, X, y, criterion):
    preds = model(X)
    loss = criterion(preds, y)
    return loss.item()


def get_accuracy(model, X, y):
    preds = model(X)
    correct = 0
    total = 0
    for pred, yb in zip(preds, y):
        pred = int(torch.argmax(pred))
        yb = int(torch.argmax(yb))
        if pred == yb:
            correct += 1
        total += 1
    acc = round(correct / total, 3) * 100
    return acc
