import matplotlib.pyplot as plt

import torch
import torchvision
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau


def main():
    data_dir = 'Cat_Dog_data'

    Debug = False
    if Debug: 
        print("Defining the trasnforms: ")
        print("   - Resize: 255")
        print("   - CenterCrop: 224")
        print("   - ToTensor")
    transform = transforms.Compose([transforms.Resize(255),        # If size is an int, smaller edge of the image will be matched to this number
                                    transforms.CenterCrop(224),    # Crops the given image at the center
                                    transforms.ToTensor()])        # Convert a PIL Image or ndarray to tensor and scale the values accordingly

    if Debug: print("Create the ImageFolder") 
    dataset = datasets.ImageFolder(data_dir, transform=transform) # Data loader for images

    if Debug: print("Create the DataLoader")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    # dataloader is a generator, images can be accessed through it using a loop

    if Debug:
        print("Augmenting data with transforms")
        print("   - RandomRotation")
        print("   - RandomResizedCrop")
        print("   - RandomHorizontalFlip")
        print("   - ToTensor")
        print("   - Normalize")
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor()],
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])]) 

    test_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
    # Test data should only be resized and center-cropped 

    if Debug: print("Applying trasnforms")
    train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
    test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)

    
    if False: # inspect the data to check wheter or not the labels are correctctly assigned
        # Get class names from the train_data
        class_names = train_data.classes
        print(f"Class names: {class_names}")

        # Visualize some training data with labels
        visualize_data(trainloader, class_names)


    if Debug: print("Defining the model")
    model = ClassifierNN()
    print(model)
    if Debug: print("Criterion: BCEWithLogitsLoss")
    if Debug: print("Optimizer: Adam")
    criterion = nn.BCEWithLogitsLoss()  # Use BCEWithLogitsLoss for binary classification
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    if Debug: print("Learning Rate scheduler: ReduceLROnPlateau")
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10) #Implement a Learning Rate Scheduler

    epochs = 5
    print("Start training")
    for e in range(epochs):
        running_loss = 0
        model.train()  # set to training mode
        for images, labels in trainloader:     
            labels = labels.float().unsqueeze(1)  # Convert labels to float and add channel dimension for BCEWithLogitsLoss
            model_output = model(images)
            loss = criterion(model_output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
        
        # Print the average loss for the epoch
        training_loss = running_loss / len(trainloader)
        validation_loss, validation_accuracy = validate(model, testloader, criterion)
        print(f"Epoch {e+1}/{epochs} - Training loss: {training_loss:.4f} - Validation loss: {validation_loss:.4f} - Validation accuracy: {validation_accuracy:.2f}%")

        # Step the scheduler with the current epoch validation loss
        scheduler.step(validation_loss)
        


def validate(model, testloader, criterion):
    model.eval() # set to eval mode
    test_loss = 0
    accuracy = 0
    with torch.no_grad():
        for images, labels in testloader:
            labels = labels.float().unsqueeze(1)  # Convert labels to float and add channel dimension for BCEWithLogitsLoss
            output = model(images)
            test_loss += criterion(output, labels).item()
            ps = torch.exp(output)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    return test_loss / len(testloader), accuracy * 100 / len(testloader)



class ClassifierNN(nn.Module):
    def __init__(self):
        super(ClassifierNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)  # input channels, output channels, kernel size
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)  # Increase dropout rate to 50%
        self.bn1 = nn.BatchNorm2d(16)   # Add batch normalization
        self.bn2 = nn.BatchNorm2d(32)   # normalizes the output of a previous activation layer
        self.fc1 = nn.Linear(32 * 56 * 56, 256) 
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1) # Single output unit for binary classification
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56) # Flatten the tensor before passing to the fully connected layer || x = x.view(x.shape[0], -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        #x = F.log_softmax(x, dim=1)
        # No activation function here since BCEWithLogitsLoss expects logits

        return x

def visualize_data(dataloader, class_names):
    data_iter = iter(dataloader)
    images, labels = next(data_iter)  # Get the next batch of data
    print(f"Images batch shape: {images.size()}")
    print(f"Labels batch shape: {labels.size()}")

    # Display images
    plt.figure(figsize=(10, 10))
    grid_img = torchvision.utils.make_grid(images, nrow=4)
    plt.imshow(grid_img.permute(1, 2, 0))

    # Print labels
    for i in range(len(labels)):
        print(f"Label {i}: {class_names[labels[i]]}")
    plt.show()

if __name__ == "__main__":
   main()