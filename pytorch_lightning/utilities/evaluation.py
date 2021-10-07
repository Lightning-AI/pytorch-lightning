import torch


def get_loss(model, X, y, criterion, device="cpu", model_eval=False) -> float:
    """
    model:
    X:Inputs of the Model
    y:Ground Truths
    criterion:
    device:the device that the model and all of the operations are run on
    model_eval:should this funtion convert the model to a train state or eval state
    """
    if model_eval is True:  # Check is model_eval is True
        model.eval()  # If model_eval is true this funtion will turn the model's state to evaluation.
    else:
        model.train()  # If model_eval is false this funtion will turn the model's state to train
    model.to(device)
    X = X.to(device)
    y = y.to(device)
    preds = model(X)  # Predicting X
    loss = criterion(preds, y)  # Calculating loss
    return loss.item()


def get_accuracy(model, X, y, device: str = "cpu", model_eval: bool = False, argmax: bool = False) -> float:
    """
    model:
    X:Inputs of the Model
    y:Ground Truths
    criterion:
    device:the device that the model and all of the operations are run on
    model_eval:should this funtion convert the model to a train state or eval state
    argmax:True - [0,1,0] [1,0,0] False -1 5
    """
    if model_eval is True:  # Check is model_eval is True
        model.eval()  # If model_eval is true this funtion will turn the model's state to evaluation.
    else:
        model.train()  # If model_eval is false this funtion will turn the model's state to train
    model.to(device)
    X = X.to(device)
    y = y.to(device)
    preds = model(X)  # Predicting X
    preds = preds.to(device)  # Convert predictions to the device
    correct = 0
    total = 0
    for pred, yb in zip(preds, y):  # iterating over the predictions and the truth
        if argmax:  # check if the pred and the truth is like [0,1,0] [1,0,0]
            pred = int(torch.argmax(pred))
            yb = int(torch.argmax(yb))
        else:  # checking if the pred and the truth is like 1 5
            pred = int(torch.round(pred))
            yb = int(torch.round(yb))
        if pred == yb:
            correct += 1
        total += 1
    acc = round(correct / total, 3) * 100
    return acc
