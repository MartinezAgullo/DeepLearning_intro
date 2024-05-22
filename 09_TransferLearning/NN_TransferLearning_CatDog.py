
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import matplotlib.pyplot as plt

from torchvision.models import DenseNet121_Weights



def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("Using GPU.")
    else:
        print("CUDA not available. Training on CPU.")

    data_dir = '../08_LoadingAndSavingModels/Cat_Dog_data'

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])


    train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
    test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)

    if False: # inspect the data to check wheter or not the labels are correctctly assigned
        # Get class names from the train_data
        class_names = train_data.classes
        print(f"Class names: {class_names}")

        # Visualize some training data with labels
        visualize_data(trainloader, class_names)

    print("Import model: densenet121")
    model = models.densenet121(weights=DenseNet121_Weights.DEFAULT)
    #print(model)

    # The "features" part of densenet121 is usefull, so freeze it
    for param in model.parameters():
        param.requires_grad = False
        
    # The "classifer" fully-connected layer has to be 
    # modified to work for binary classification
    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(1024, 500)), #1024 input features. 500 input in hidden layer
                            ('relu', nn.ReLU()),
                            ('fc2', nn.Linear(500, 2)),
                            ('output', nn.LogSoftmax(dim=1))
                            ]))

    model.classifier = classifier  # Attach the new classifier to our model

    criterion = nn.NLLLoss()
    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    model.to(device)

    print("Start training")
    epochs = 3
    steps = 0
    running_loss = 0
    print_every = 5
    for epoch in range(epochs):                 # Loop on epochs
        for inputs, labels in trainloader:      # Loop on data
            steps += 1
            # Move input and label tensors to the default device (GPU if availabre)
            inputs, labels = inputs.to(device), labels.to(device)

            logps = model.forward(inputs)          # Forward Pass: Compute the output of the model || logps is a Log Probability
            loss = criterion(logps, labels)        # Calculate the loss using the model's output and the true labels

            optimizer.zero_grad()   # Reset gradient
            loss.backward()         # Backpropagate :: Calculate gradients
            optimizer.step()        # Update model parameters

            running_loss += loss.item()     # Keep track un loos

            if steps % print_every == 1:    # Test model
                test_loss = 0
                accuracy = 0
                model.eval()  # Set the model to evaluation mode // Use it to make predictions
                with torch.no_grad():
                    for inputs, labels in testloader:   # Access test data
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)   # Model restunds log soft max :: Log Probabilities
                        batch_loss = criterion(logps, labels)
                        
                        test_loss += batch_loss.item()
                        
                        # Calculate accuracy
                        ps = torch.exp(logps)                                   # Get Probability from Log Probabilities
                        top_p, top_class = ps.topk(1, dim=1)                    # Get the class with the largest probability
                        equals = top_class == labels.view(*top_class.shape)     # Check for equality with the labels
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item() # Use the aquality to calculate the accuracy
                        
                print(f"Epoch {epoch+1}/{epochs}.. "
                    f"Train loss: {running_loss/print_every:.3f}.. "        # Average training loss
                    f"Test loss: {test_loss/len(testloader):.3f}.. "        # Average among all batches
                    f"Test accuracy: {accuracy/len(testloader):.3f}")       # Average among all batches
                running_loss = 0
                model.train() # Set the model back to training mode


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