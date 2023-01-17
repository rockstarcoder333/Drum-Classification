import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F

class EarlyStopping:
    def __init__(self, patience=5, threshold=0.01):
        self.patience = patience
        self.threshold = threshold
        self.best_loss = None
        self.counter = 0

    def validate(self, loss):
        if self.best_loss is None:
            self.best_loss = loss
            return False
        elif loss - self.best_loss > self.threshold:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_loss = loss
            self.counter = 0
        return False


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=(3, 3), padding='same')
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(5, 5), padding='same')
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.fc1 = nn.Linear(32 * 8 * 8, 100)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 32 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def prepare_dataset(train_folder, validation_folder, test_folder):
    train_transform = transforms.Compose([transforms.Resize((32, 32)),
                                        transforms.RandomRotation(3),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    test_transform = transforms.Compose([transforms.Resize((32, 32)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    train_data = datasets.ImageFolder(train_folder, transform=train_transform)
    validation_data = datasets.ImageFolder(validation_folder, transform=test_transform)
    test_data = datasets.ImageFolder(test_folder, transform=test_transform)

    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    validation_loader = DataLoader(validation_data, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

    return train_loader, validation_loader, test_loader

def train_new_model(model_path, train_loader, validation_loader, test_loader):
    model = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    early_stopping = EarlyStopping(patience=2)
    n_epochs = 30
    best_validation_loss = float('inf')
    for epoch in range(1, n_epochs+1):
        running_loss = 0.0
        running_accuracy = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predictions = torch.max(outputs, 1)
            running_accuracy += torch.mean((predictions == labels).type(torch.FloatTensor))
        else:
            validation_loss = 0.0
            accuracy = 0.0
            model.eval()
            with torch.no_grad():
                for images, labels in validation_loader:
                    log_ps = model(images)
                    validation_loss += criterion(log_ps, labels)
                    ps = torch.exp(log_ps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor))
                if early_stopping.validate(validation_loss/len(validation_loader)):
                    break
                if validation_loss < best_validation_loss:
                    print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                    best_validation_loss,
                    validation_loss))
                    best_validation_loss = validation_loss
                    torch.save(model.state_dict(), model_path)
            model.train()
            print("Epoch: {}/{}".format(epoch, n_epochs),
                  "Training Loss: {:.3f}".format(running_loss/len(train_loader)),
                  "Training Accuracy: {:.3f}".format(running_accuracy/len(train_loader)),
                  "Validation Loss: {:.3f}".format(validation_loss/len(validation_loader)))

    model.load_state_dict(torch.load(model_path))
    model.eval()
    test_loss = 0.0
    accuracy = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            print("======", images[0], labels)
            log_ps = model(images)
            test_loss += criterion(log_ps, labels)
            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))
    print("Test Loss: {:.3f}.. ".format(test_loss/len(test_loader)),
          "Test Accuracy: {:.3f}".format(accuracy/len(test_loader)))

if __name__ == '__main__':
    model_path = "model.pt"
    train_folder = "train"
    validation_folder = "validation"
    test_folder = "test"

    train_loader, validation_loader, test_loader = prepare_dataset(train_folder, validation_folder, test_folder)
    train_new_model(model_path, train_loader, validation_loader, test_loader)
