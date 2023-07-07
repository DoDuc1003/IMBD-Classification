from tqdm import tqdm
import torch
def train_model(model, dataloader, criterion, optimizer, epoch, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    running_loss = 0.0
    for i, (batch_inputs, batch_labels, batch_masks) in tqdm(enumerate(dataloader)):

        batch_inputs = batch_inputs.to(device)
        batch_labels = batch_labels.to(device)
        batch_masks = batch_masks.to(device)

        optimizer.zero_grad()

        outputs = model(batch_inputs, batch_masks)
        loss = criterion(outputs, batch_labels)

        loss.backward()
        optimizer.step()

        batch_inputs = batch_inputs.to('cpu')
        batch_labels = batch_labels.to('cpu')
        batch_masks = batch_masks.to('cpu')

        running_loss += loss.item() * batch_inputs.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    
    print("saving model")
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
    
    torch.save(model.state_dict(), f'model{i}.pth')
    print("Training complete epoch!")

from sklearn.metrics import classification_report
def test_model(model, dataloader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()
    total_loss = 0

    y_true = []
    y_pred = []
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for i, (batch_inputs, batch_labels, batch_masks) in tqdm(enumerate(dataloader)):
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)
            batch_masks = batch_masks.to(device)

            outputs = model(batch_inputs, batch_masks)

            loss = criterion(outputs, batch_labels)

            total_loss += loss.item()
            
            _, predicted = torch.max(outputs, dim=1)

            y_pred = y_pred + predicted.to('cpu').tolist()
            y_true = y_true + batch_labels.to('cpu').tolist()

            batch_inputs = batch_inputs.to('cpu')
            batch_labels = batch_labels.to('cpu')
            batch_masks = batch_masks.to('cpu')
    report = classification_report(y_true, y_pred)
    with open('out.txt', 'a') as file:
        file.write(report + '\n')
    print(report)