# Load in relevant libraries, and alias where appropriate
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.ao.quantization as tq
import matplotlib.pyplot as plt
import torch.nn.functional as F

    
# Define relevant variables for the ML task
batch_size = 64
num_classes = 10  #number of output classes (10 for MNIST)
learning_rate = 0.001
num_epochs = 30
early_stop_threshold = 5
    
# Device will determine whether to run the training on GPU or CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Binarize(object):
    """Convert a tensor with values in [0,1] to {0,1} by thresholding."""
    def __init__(self, thresh: float = 0.5):
        self.thresh = thresh
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        # tensor is C×H×W, already float32 after ToTensor
        return (tensor > self.thresh).float()


#defining the quantized convolutional layer
class QuantConv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, num_bits):
      super().__init__(in_channels, out_channels, kernel_size, stride, padding)  #initializes the parent object 
      self.num_bits = num_bits
  
    def forward(self, x):
      q_w = QuantizedWeightSTE.apply(self.weight, self.num_bits) #q_w is the quantized weights of the layer
      return F.conv2d(x, q_w, self.bias, self.stride,
                      self.padding, self.dilation, self.groups) #applies a 2d conv over the input with quantized weights
   
#defining the quantized linear layer
class QuantLinear(nn.Linear):
    def __init__(self, in_features, out_features, num_bits): 
      super().__init__(in_features, out_features, bias=False) #initializes the parent object
      self.num_bits = num_bits
  
    def forward(self, x):
      q_w = QuantizedWeightSTE.apply(self.weight, self.num_bits) #q_w is the quantized weights of the layer
      return F.linear(x, q_w) #applies a linear transform on the input with quantized weights

#Defining the convolutional neural network
class LinearModel(nn.Module):  #class inherits fromm nn.Module, the base class for all Pytorch models
    def __init__(self, num_classes):  #constructor initializes layers & takes num_classes as input 
        super(LinearModel, self).__init__() #calls constructor of nn.Module (necessary for setup)
        
        self.fc1 = QuantLinear(784, 128, num_bits=16)
        self.sigmoid1 = QuantizedMish() #apply quantized sigmoid activation to output
        self.fc2 = QuantLinear(128, num_classes, num_bits=16)

    def forward(self, x):
        out = x.reshape(x.size(0), -1) #flatten input to [batch_size, 784]
        out = self.fc1(out)
        out = self.sigmoid1(out)
        out = self.fc2(out)
        return out          #nn.CrossEntropyLoss applies softmax and negative log likelihood loss
    
#applies quantization with a straight-through estimator in the backward pass to ensure gradients flow on the standard input
class QuantizedActivationSTE(torch.autograd.Function):
  @staticmethod
  def forward(ctx, x):
    mul1_fn = torch.ao.nn.quantized.FloatFunctional()
    
    x_q = torch.floor(mul1_fn.mul(x, 2.0)) #quantize to integer range [0, 2]
    x_q = torch.clamp(x_q, 0, 1) #make sure it is in the integer range [0, 1]

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

    self.mish = nn.Sigmoid()

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
  full_train_dataset = torchvision.datasets.MNIST(root = './data',
                                                train = True,
                                                transform = transforms.Compose([
                                                        transforms.Resize((28,28)),
                                                        transforms.ToTensor(),
                                                        Binarize(0.5)]),
                                                download = True)

  #split full_train_dataset into training set and validation set
  train_size = int(0.83 * len(full_train_dataset)) #50k for training
  val_size = len(full_train_dataset) - train_size #10k for validation
  train_dataset, _ = torch.utils.data.random_split(full_train_dataset, [train_size, val_size])

  #create a separate validation dataset with the test transforms
  val_dataset = torchvision.datasets.MNIST(root = './data',
                                                train = True,
                                                transform = transforms.Compose([
                                                        transforms.Resize((28,28)),
                                                        transforms.ToTensor(),
                                                        Binarize(0.5)]),
                                                download = True)

  #subset only the remaining 10% of the data for validation
  _, val_dataset = torch.utils.data.random_split(val_dataset, [train_size, val_size])
      
      
  test_dataset = torchvision.datasets.MNIST(root = './data',
                                                train = False,
                                                transform = transforms.Compose([
                                                        transforms.Resize((28,28)),
                                                        transforms.ToTensor(),
                                                        Binarize(0.5)]),
                                                download=True)
      
      
  train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                                batch_size = batch_size,
                                                shuffle = True)
  val_loader = torch.utils.data.DataLoader(dataset = val_dataset,
                                                batch_size = batch_size,
                                                shuffle = True)
      
  test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                                batch_size = batch_size,
                                                shuffle = True)
  model = LinearModel(num_classes).to(device)
      
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


      #check for early stopping based on validation loss
      if val_losses[-1] < best_val_loss:   #index -1 returns element at end of val_losses
        best_val_loss = val_losses[-1]
        epochs_without_improvement = 0
      else:
        epochs_without_improvement += 1

      # If the validation loss starts increasing or is plateauing, stop training
      if epochs_without_improvement >= early_stop_threshold:
        print(f"Early stopping at epoch {epoch + 1}")
        break

      #print val statistics after each epoch
      print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

      #save model with best val_accuracy
      if val_accuracy >= best_val_accuracy:
        print(f"Best Accuracy: {best_val_accuracy:.2f}%, Val Accuracy: {val_accuracy:.2f}%")
        torch.save(model.state_dict(), './ComplexLinearModel_weights')
        best_val_accuracy = val_accuracy
        print(f"Model updated at epoch {epoch + 1}")

  print(f"Best Train Loss {min(loss_history):.4f}")
  print(f"Best Val Loss {min(val_losses):.4f}")

  plt.figure(figsize=(6,4))
  plt.plot(range(1, epoch + 2), loss_history, marker="o")
  plt.title("Training loss per epoch")
  plt.xlabel("Epoch")
  plt.ylabel("Cross-entropy loss")
  plt.grid(True)
  plt.tight_layout()
  plt.savefig("loss_graph_complex.png", dpi=300)  # Save as an image

  # Test the model
  # In the test phase, we don't need to compute gradients (for memory efficiency)

  PATH = "./ComplexLinearModel_weights"
  model.load_state_dict(torch.load(PATH)) #load weights/parametes of best trained model into model 
  model.eval()  # Set the model to evaluation mode

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
  
if __name__ == "__main__":
    main()