from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import torch


def test_model(model, test_loader, device):
    correct = 0
    total = 0
    ce_loss = 0
    with torch.no_grad():
        for imgs, labels in test_loader:

            imgs = imgs.to(device)
            labels = labels.to(device)

            preds = model(imgs)
            _, answers = torch.max(preds.data, 1)

            total += labels.size(0)
            correct += (answers == labels).sum().item()

            ce_loss += F.cross_entropy(preds, labels).item()
    ce_loss /= len(test_loader)
    accuracy = correct / total
    return ce_loss, accuracy


def train_hard_labels(model, train_loader, val_loader, num_epochs, lr, 
                      device, log=False):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    val_acc = []
    for i in range(num_epochs):

        if log:
            print("Epoch {}".format(i))

        running_loss = 0.0
        total = 0
        correct = 0

        for i_batch, (imgs, labels) in enumerate(train_loader):

            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            pred = model(imgs)

            _, answers = torch.max(pred.data, 1)
            total += labels.size(0)
            correct += (answers == labels).sum().item()

            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_acc = correct/total
        ce_loss, acc = test_model(model, val_loader, device)

        val_acc.append(acc)

        if log:
            print("Train: accuracy: {:.3f}, CE: {:.3f}".format(train_acc, running_loss/len(train_loader)))
            print("Validation: accuracy: {:.3f}, CE: {:.3f}".format(acc, ce_loss))

    return val_acc


def distill_loss(distill_pred, soft_labels, labels, T, alpha):
    hard_loss = F.cross_entropy(distill_pred, labels)

    distill_probs = F.log_softmax(distill_pred/T, dim=1)
    soft_probs = F.log_softmax(soft_labels/T, dim=1)

    soft_loss = F.kl_div(distill_probs, soft_probs, reduction='batchmean')

    return alpha * soft_loss * (T**2) + (1 - alpha) * hard_loss


def train_soft_labels(distill_model, teacher_model, train_loader, val_loader, 
                      num_epochs, lr, T, alpha, device, log=False):
    """
    distill_model - модель, которую обучаем
    teacher_model - модель, которая генерирует "мягкие" ответы
    T - температура
    alpha - коэфициент перед ошибкой:
        loss = alpha * soft_loss + (1 - alpha) * hard_loss
    """

    optimizer = optim.Adam(distill_model.parameters(), lr=lr)

    val_acc = []
    for i in range(num_epochs):

        if log:
            print("Epoch {}".format(i))

        total = 0
        correct = 0
        running_loss = 0.0
        for i_batch, (imgs, labels) in enumerate(train_loader):
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            distill_pred = distill_model(imgs)
            soft_labels = teacher_model(imgs).detach()

            loss = distill_loss(distill_pred, soft_labels, labels, T, alpha)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, answers = torch.max(distill_pred.data, 1)
            total += labels.size(0)
            correct += (answers == labels).sum().item()

        train_acc = correct/total
        ce_loss, acc = test_model(distill_model, val_loader)

        val_acc.append(acc)

        if log:
            print("Train: accuracy: {:.3f}, CE: {:.3f}".format(train_acc, running_loss/len(train_loader)))
            print("Validation: accuracy: {:.3f}, CE: {:.3f}".format(acc, ce_loss))

    return val_acc
