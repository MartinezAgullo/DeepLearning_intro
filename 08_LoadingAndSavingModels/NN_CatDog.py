import matplotlib.pyplot as plt

import torch
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau


def main():
    data_dir = 'Cat_Dog_data'

    print("Defining the trasnforms: ")
    print("   - Resize: 255")
    print("   - CenterCrop: 224")
    print("   - ToTensor")
    transform = transforms.Compose([transforms.Resize(255),        # If size is an int, smaller edge of the image will be matched to this number
                                    transforms.CenterCrop(224),    # Crops the given image at the center
                                    transforms.ToTensor()])        # Convert a PIL Image or ndarray to tensor and scale the values accordingly

    print("Create the ImageFolder") 
    dataset = datasets.ImageFolder(data_dir, transform=transform) # Data loader for images

    print("Create the DataLoader")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    # dataloader is a generator, images can be accessed through it using a loop


    print("Augmenting data with transforms")
    print("   - RandomRotation")
    print("   - RandomResizedCrop")
    print("   - RandomHorizontalFlip")
    print("   - ToTensor")
    print("   - Resize")
    print("   - CenterCrop")
    print("   - ToTensor")
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor()]) 

    test_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor()])
    # Test data should only be resized and center-cropped 

    print("Applying trasnforms")
    train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
    test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=32)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)

    print("Defining the model")
    model = ClassifierNN()
    print(model)
    print("Criterion: NLLLoss")
    print("Optimizer: Adam")
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    print("Using Learning Rate scheduler")
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10) #Implement a Learning Rate Scheduler

    epochs = 5
    print("Start training")
    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:            
            model_output = model(images)
            loss = criterion(model_output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
        
        # Print the average loss for the epoch
        epoch_loss = running_loss / len(trainloader)
        print(f"Training loss: {epoch_loss}")

        # Step the scheduler with the current epoch loss
        scheduler.step(epoch_loss)
        

class ClassifierNN(nn.Module):
    def __init__(self):
        super(ClassifierNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)  # input channels, output channels, kernel size
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)  # Dropout layer with 25% probability
        self.fc1 = nn.Linear(32 * 56 * 56, 256) 
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 2)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56) # Flatten the tensor before passing to the fully connected layer || x = x.view(x.shape[0], -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x


class ClassifierNN_old(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(150528, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 2)
        
    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.log_softmax(self.fc4(x), dim=1)
        
        return x

def defininions():
    print("A Convolutional Layer (Conv2d) is a type of layer that deals primarly with image data.  The layer applies a series of filters (kernels) to the input. Each filter slides across the width and height of the input image, computing dot products between the filter and input at each position. This process effectively extracts features (like edges, shapes, and textures) from the input image, producing a feature map that emphasizes important features, making it very effective for image processing tasks.")
    print("self.conv1 = nn.Conv2d(3, 16, 3, padding=1)   Arguments:")
    print("   - First Argument (3): This specifies the number of input channels that the layer expects. For color images (such as those in RGB format), this is typically 3")
    print("   - Second Argument (16): This indicates the number of output channels, or how many different filters the convolutional layer will apply to the input")
    print("   - Third Argument (3): This is the size of the filter (or kernel) that will be used, which is 3x3 in this case")
    print("Conv2d does not need the explicit input size because convolutional operations are defined to work with any input size that satisfies the kernel, padding, and stride requirements")
    
    print("A Linear Layer (often called a fully connected or dense layer) connects every input to every output by a learned linear transformation. It is defined by two main parameters (weights and biases). This layer's primary purpose is to map the learned features (from the previous layers) to the desired output size.")
    
    print("The MaxPool2d layer is a form of pooling layer that performs downsampling by dividing the input into rectangular pooling regions and computing the maximum value of each region. It is used in CNNs to reduce the dimensions of the feature map")

if __name__ == "__main__":
   main()