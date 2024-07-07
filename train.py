import torch

def batch_train(model, device, train_loader, val_loader, optimizer, criterion, scheduler): 

    # Model in train
    model.train()
    running_loss = 0.0
    correct = 0
    size = len(train_loader.dataset)

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        correct += (output.argmax(1) == target).type(torch.float).sum().item()
        running_loss += loss.item()

    acc = (100*correct)/size
    avg_loss = running_loss/(batch_idx+1)
    print(f"Train: Avg loss: {avg_loss:>8f}, Accuracy: {(acc):>0.2f}%")

    # Model in validation
    model.eval()
    running_loss = 0.0
    correct = 0
    size = len(val_loader.dataset)

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output, target)
            running_loss += loss.item()
            correct += (output.argmax(1) == target).type(torch.float).sum().item()

    vacc = (100*correct)/size
    avg_vloss = running_loss/(batch_idx+1)
    print(f"Validation: Avg loss: {avg_vloss:>8f}, Accuracy: {(vacc):>0.2f}%")
    scheduler.step()

    return avg_loss, avg_vloss, acc, vacc