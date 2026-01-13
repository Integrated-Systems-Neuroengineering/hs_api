import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.ao.quantization as tq
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os

from spikingjelly.activation_based import neuron, functional, surrogate, layer
from copy import deepcopy
'''
28x28 input resolution
Adapted from /LeNet5/QAT_LeNet5Model_NoPooling_2conv_BinarySigmoidnewThres

Binary Sigmoid is > thres
Best Model determined based on lowest validation loss
Epoch patience resets to 0 if loss improves

Convolutional layers have bias=False
BatchNorm2d removed

Uses 2-channel MNIST, same dataset for both channels, but different thresholds
'''
# Define relevant variables for the ML task
batch_size = 64
num_classes = 10  #number of output classes (10 for MNIST)
learning_rate = 0.001
num_epochs = 100000
early_stop_threshold = 5

PATH = "/home/k7arora/hs_api/examples/CRI_Mapping/chris_code/converter_testing/2Channel_MNIST/" #path for saving weights
    
# Device will determine whether to run the training on GPU or CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Binarize(object):
    """Convert a tensor with values in [0,1] to {0,1} by thresholding."""
    def __init__(self, thresh: float = 0.5):
        self.thresh = thresh
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor > self.thresh).float()

#defining the quantized convolutional layer
class QuantConv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, num_bits):
      super().__init__(in_channels, out_channels, kernel_size, stride, padding, bias=False)  #initializes the parent object, Conv2d
      self.num_bits = num_bits
  
    def forward(self, x):
      q_w = QuantizedWeightSTE.apply(self.weight, self.num_bits) #q_w is the quantized weights of the layer
      return F.conv2d(x, q_w, self.bias, self.stride,
                      self.padding, self.dilation, self.groups) #applies a 2d conv over the input with quantized weights
   
#defining the quantized linear layer
class QuantLinear(nn.Linear):
    def __init__(self, in_features, out_features, num_bits): 
      super().__init__(in_features, out_features, bias=False) #initializes the parent object, nn.Linear
      self.num_bits = num_bits
  
    def forward(self, x):
      q_w = QuantizedWeightSTE.apply(self.weight, self.num_bits) #q_w is the quantized weights of the layer
      return F.linear(x, q_w, self.bias) #applies a linear transform on the input with quantized weights

#keli's DVSGestureNet from original paper, with ability to change encoder
class DVSGestureNetNoBias(nn.Module):
    def __init__(self, channels=128, encoder = 3, out_features = 512, spiking_neuron: callable = None, input_shape = (16, 2, 128, 128), **kwargs):
        super().__init__()

        B, C, H, W = input_shape

        conv = []
        for i in range(encoder):
            if conv.__len__() == 0:
                in_channels = 2
            else:
                in_channels = channels

            if H > 3 and W > 3: #don't want to reduce spatial dims to 1x1 which fails batchnorm
                conv.append(layer.Conv2d(in_channels, channels, kernel_size=3, stride = 2, padding=0, bias=False))
                conv.append(layer.BatchNorm2d(channels))
                conv.append(spiking_neuron(**deepcopy(kwargs)))
                H = H // 2
                W = W // 2
            
            else:
                conv.append(layer.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False))
                conv.append(layer.BatchNorm2d(channels))
                conv.append(spiking_neuron(**deepcopy(kwargs)))
            
            print(H)
            print(W)

        conv_seq = nn.Sequential(*conv)
        #print(conv_seq)
        B, C, H, W = input_shape
        dummy_input = torch.zeros((B, C, H, W))

        with torch.no_grad():
            x_out = conv_seq(dummy_input)
            #flatten but ignore batch size
            in_features = x_out.flatten(start_dim=1).shape[1]  # Flatten and get feature dim
            print(x_out.shape)

        print("Input features to first linear layer:", in_features)    

        self.conv_fc = nn.Sequential(
            *conv,
            
            layer.Flatten(),
            layer.Dropout(0.5), #default 0.5
            layer.Linear(in_features, out_features, bias=False),
            spiking_neuron(**deepcopy(kwargs)),

            layer.Dropout(0.5), #default 0.5
            layer.Linear(out_features, 11, bias=False),
            spiking_neuron(**deepcopy(kwargs)),

        )

    def forward(self, x: torch.Tensor):
        return self.conv_fc(x)
#applies quantization with a straight-through estimator in the backward pass to ensure gradients flow on the standard input
class QuantizedActivationSTE(torch.autograd.Function):
  @staticmethod
  def forward(ctx, x):
    mul1_fn = torch.ao.nn.quantized.FloatFunctional()
    
    epsilon = 1e-6  # small constant to exclude exact 0.5
    x_q = torch.floor(mul1_fn.mul(x - epsilon, 2.0)) # now 0.5 becomes 0.999999 → floor → 0
    x_q = torch.clamp(x_q, 0, 1) # ensure result is still in {0, 1}

    ctx.save_for_backward(x) #save input for backward

    return x_q

  @staticmethod
  def backward(ctx, grad_output):
    #backward pass: Use STE (pass gradients as if quantization didn't exist)
    grad_input = grad_output.clone()  #pass gradient straight through
    return grad_input, None #none needed to bypass num_bits

#mish activation that uses the quantization activation STE for smooth gradients
class QuantizedMish(nn.Module):
  def __init__(self):
    super(QuantizedMish, self).__init__()

    self.mish = nn.Sigmoid()  #define activation function as sigmoid

  def forward(self, x):
    x = self.mish(x)
    x = QuantizedActivationSTE.apply(x) #quantize
    return x
  
#applies quantization with a straight-through estimator in the backward pass to ensure gradients flow on the standard input for weights (symmetric)
class QuantizedWeightSTE(torch.autograd.Function):
  @staticmethod
  def forward(ctx, x, num_bits):
    #quantize to number of bits, keeping scale the same
    max = torch.max(torch.abs(x)) #max abs val
    max = torch.where(max == 0, torch.tensor(1.0), max) #avoid dividing by 0 if max is 0

    levels = 2**(num_bits-1)-1 #number of quantization levels
    x_q = torch.round(x / max * levels) #quantize to range [-(2**(num_bits-1)-1),  2**(num_bits-1)-1], integers, max becomes levels

    x_quantized_scaled = x_q * max / levels #scale back to the range the input was in (all positive floating point, max is preserved)

    ctx.save_for_backward(x) #save input for backward

    return x_quantized_scaled

  @staticmethod
  def backward(ctx, grad_output):
    #backward pass: Use STE (pass gradients as if quantization didn't exist)
    grad_input = grad_output.clone()  #pass gradient straight through
    return grad_input, None


def main():
  #Loading the dataset and preprocessing
  #krish: changed size to 90x90, added two channels(duplicates except for diff thresholds) to mimic DVSGesture dataset
  #first channel has 0.5 threshold
  full_train_dataset1 = torchvision.datasets.MNIST(root = './data',
                                                train = True,
                                                transform = transforms.Compose([
                                                        transforms.Resize((90,90)), 
                                                        transforms.ToTensor(),
                                                        Binarize(0.5)]),
                                                download = True)
  
  #second channel has 0.25 threshold
  full_train_dataset2 = torchvision.datasets.MNIST(root = './data',
                                                train = True,
                                                transform = transforms.Compose([
                                                        transforms.Resize((90,90)), 
                                                        transforms.ToTensor(),
                                                        Binarize(0.25)]),
                                                download = True)



  #split full_train_dataset into training set and validation set
  train_size = int(0.83 * len(full_train_dataset1)) #50k for training
  val_size = len(full_train_dataset1) - train_size #10k for validation
  train_dataset1, _ = torch.utils.data.random_split(full_train_dataset1, [train_size, val_size])
  train_dataset2, _ = torch.utils.data.random_split(full_train_dataset2, [train_size, val_size])

  #create a separate validation dataset with the test transforms
  #first channel has 0.5 threshold
  val_dataset1 = torchvision.datasets.MNIST(root = './data',
                                                train = True,
                                                transform = transforms.Compose([
                                                        transforms.Resize((90,90)),
                                                        transforms.ToTensor(),
                                                        Binarize(0.5)]),
                                                download = True)
  
  #second channel has 0.25 threshold
  val_dataset2 = torchvision.datasets.MNIST(root = './data',
                                                  train = True,
                                                  transform = transforms.Compose([
                                                          transforms.Resize((90,90)),
                                                          transforms.ToTensor(),
                                                          Binarize(0.25)]),
                                                  download = True)

  #subset only the remaining 10% of the data for validation
  _, val_dataset1 = torch.utils.data.random_split(val_dataset1, [train_size, val_size])
  _, val_dataset2 = torch.utils.data.random_split(val_dataset2, [train_size, val_size])

  #first channel has 0.5 threshold
  test_dataset1 = torchvision.datasets.MNIST(root = './data',
                                                train = False,
                                                transform = transforms.Compose([
                                                        transforms.Resize((90,90)),
                                                        transforms.ToTensor(),
                                                        Binarize(0.5)]),
                                                download=True)

  #second channel has 0.25 threshold
  test_dataset2 = torchvision.datasets.MNIST(root = './data',
                                                train = False,
                                                transform = transforms.Compose([
                                                        transforms.Resize((90,90)),
                                                        transforms.ToTensor(),
                                                        Binarize(0.25)]),
                                                download=True)


  # Helper function to stack two Subset datasets along channel dimension
  def stack_subsets(subset1, subset2):
    # Both subsets have the same indices and length
    stacked_images = []
    stacked_labels = []
    for i in range(len(subset1)):
        img1, label1 = subset1[i]
        img2, label2 = subset2[i]
        # Stack along channel dimension
        stacked_img = torch.cat([img1, img2], dim=0)  # [2, H, W]
        stacked_images.append(stacked_img)
        stacked_labels.append(label1)  # labels should be the same
    images_tensor = torch.stack(stacked_images)
    labels_tensor = torch.tensor(stacked_labels)
    return torch.utils.data.TensorDataset(images_tensor, labels_tensor)

  train_dataset = stack_subsets(train_dataset1, train_dataset2)
  val_dataset = stack_subsets(val_dataset1, val_dataset2)
  test_dataset = stack_subsets(test_dataset1, test_dataset2)

  #print info about datasets, shape should be [2, 90, 90]
  print("Training dataset size: ", len(train_dataset), ", shape: ", train_dataset[0][0].shape)
  print("Validation dataset size: ", len(val_dataset), ", shape: ", val_dataset[0][0].shape)
  print("Test dataset size: ", len(test_dataset), ", shape: ", test_dataset[0][0].shape)


  train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                                batch_size = batch_size,
                                                shuffle = True)
  val_loader = torch.utils.data.DataLoader(dataset = val_dataset,
                                                batch_size = batch_size,
                                                shuffle = True)
      
  test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                                batch_size = batch_size,
                                                shuffle = True)
      
  model = DVSGestureNetNoBias(
      channels=4,
      encoder=2,
      spiking_neuron=neuron.IFNode,
      surrogate_function=surrogate.ATan(),
      input_shape=(batch_size, 2, 90, 90),  # input shape for the model(B,C,H,W)
      detach_reset=True,
  )
      
  #Setting the loss function
  cost = nn.CrossEntropyLoss()
      
  #Setting the optimizer with the model parameters and learning rate
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
      
  #this is defined to print how many steps are remaining when training
  total_step = len(train_loader)

  #lists to store loss and accuracy values
  loss_history = []
  val_losses = []
  val_accuracies = []

  #Keeping track of best val loss and accuracy for early stop
  best_val_accuracy = 0.0
  epoch_best_val_accuracy = 0
  best_val_loss = 100.0 #absurd loss
  epochs_without_improvement = 0

  #Model Training and validation
  for epoch in range(num_epochs):
      running_loss = 0.0

      for i, (images, labels) in enumerate(train_loader):  #iterates over mini batches produced by DataLoader
          images = images.to(device)  #moves image tensor with shape [batch, channels, height, width] to computation device (CPU, GPU)
          labels = labels.to(device)  #moves label tensor with shape [batch] to computational device 
              
          #Forward pass
          outputs = model(images)  #produce logits of shape [batch, num_classes]
          loss = cost(outputs, labels) #computes cross entropy loss
          #Backward and optimize
          optimizer.zero_grad()  #clear existing gradients because Pytorch accumulates gradients by default
          loss.backward()        #fills param.grad with gradient of loss with respect to that parameter
          optimizer.step()       #updates parameter using Adam Optimizer
          running_loss += loss.item()
          if (i+1) % 400 == 0:
              print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item()))


      epoch_loss = running_loss / len(train_loader)
      loss_history.append(epoch_loss)

      #validation phase
      model.eval()  # Set the model to evaluation mode
      val_loss = 0.0
      val_correct = 0
      val_total = 0

      with torch.no_grad():
        for inputs, labels in val_loader:
          inputs, labels = inputs.to(device), labels.to(device)

          #forward pass
          outputs = model(inputs)
          loss = cost(outputs, labels)

          #statistics
          val_loss += loss.item()
          _, predicted = torch.max(outputs, 1)
          val_correct += (predicted == labels).sum().item()
          val_total += labels.size(0)

      val_loss /= len(val_loader)
      val_accuracy = 100 * val_correct / val_total

      val_losses.append(val_loss)
      val_accuracies.append(val_accuracy)

      #print val statistics after each epoch
      print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

      #check for early stopping based on validation loss
      #save model with best val_loss
      if val_losses[-1] < best_val_loss:   #index -1 returns element at end of val_losses
        print(f"Best Val Loss: {best_val_loss:.4f}, New Best Val Loss: {val_losses[-1]:.4f}")
        torch.save(model.state_dict(), os.path.join(PATH, 'QAT_DVSGestureNet_weights_NoBias_2Channels'))
        print(f"Model updated at epoch {epoch + 1}")
        best_val_loss = val_losses[-1]  #update best loss
        epochs_without_improvement = 0  #reset epochs without improvement to 0
      else:
        epochs_without_improvement += 1

      # If the validation loss starts increasing or is plateauing, stop training
      if epochs_without_improvement >= early_stop_threshold:
        print(f"Early stopping at epoch {epoch + 1}")
        break

      #update best val_accuracy
      if val_accuracy >= best_val_accuracy:
        best_val_accuracy = val_accuracy
        epoch_best_val_accuracy = epoch + 1


  #print best training and val loss. print best val accuracy
  print(f"Best Train Loss {min(loss_history):.4f}")
  print(f"Best Val Loss {min(val_losses):.4f}")
  print(f"Best Val accuracy {best_val_accuracy:.2f}% at epoch {epoch_best_val_accuracy}")

  #create loss graph
  plt.figure(figsize=(6,4))
  plt.plot(range(1, epoch + 2), loss_history, marker="o", label="Training")
  plt.plot(range(1, epoch + 2), val_losses, marker="o", linestyle="--", label="Validation")
  plt.title("Training and Validation Loss per epoch")
  plt.xlabel("Epoch")
  plt.ylabel("Cross-entropy loss")
  plt.grid(True)
  plt.legend()
  plt.tight_layout()
  plt.savefig(os.path.join(PATH, "loss_graph_QATDVSGestureNet_NoBias.png"), dpi=300)  # Save as an image

  #test phase
  model.load_state_dict(torch.load(os.path.join(PATH, "QAT_DVSGestureNet_weights_NoBias_2Channels"))) #load weights/parametes of best trained model into model
  model.eval()  # Set the model to evaluation mode

  #test model on test set 
  with torch.no_grad():
      correct = 0
      total = 0

      for images, labels in test_loader:
          images = images.to(device)
          labels = labels.to(device)
          
          outputs = model(images)
          _, predicted = torch.max(outputs.data, 1)
          
          total += labels.size(0)
          correct += (predicted == labels).sum().item()

      accuracy = 100 * correct / total
      print(f'Accuracy of the network on the 10000 test images: {accuracy:.2f} %')

      #save val and test accuracies to txt file
      with open(os.path.join(PATH, "accuracies.txt"), "w") as f:
          f.write(f"Validation Accuracy: {val_accuracy:.2f}%\n")
          f.write(f"Test Accuracy: {accuracy:.2f}%\n")

if __name__ == "__main__":
    main()