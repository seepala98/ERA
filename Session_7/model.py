import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

from tqdm import tqdm 
from torchsummary import summary

def model_summary(model, input_size):
  summary(model, input_size)

def train_model(model, train_loader, device, optimizer, train_acc, train_losses):
  model.train()
  pbar = tqdm(train_loader)
  train_loss = 0 
  correct = 0
  processed = 0
  for batch_idx, (data, target) in enumerate(pbar):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output, target)
    train_loss += loss.item()
    loss.backward()
    optimizer.step()
    correct += output.argmax(dim=1).eq(target).sum().item()
    processed += len(data)
    pbar.set_description(desc=f'loss={loss.item()}, batch_id = {batch_idx} Accuracy= {100*correct/processed:.2f}')
  
  train_acc.append(100*correct/processed)
  train_losses.append(train_loss/len(train_loader))
  return loss.item()


def test_model(model, test_loader, device, test_acc, test_losses):
  model.eval()
  test_loss = 0
  correct = 0 
  with torch.no_grad():
    for data, target in test_loader:
      data, target = data.to(device), target.to(device)
      output = model(data)
      test_loss += F.nll_loss(output, target, reduction="sum").item()
      pred = output.argmax(dim=1, keepdim=True)
      correct += pred.eq(target.view_as(pred)).sum().item()

  test_loss /= len(test_loader.dataset)
  test_acc.append(100*correct/len(test_loader.dataset))

  print('/n Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100 * correct/len(test_loader.dataset)
  ))
  test_losses.append(test_loss)
  return test_loss

class model_1_Net(nn.Module):
  def __init__(self):
    super().__init__()
    self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        )

    self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) 
    
    self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) 

    self.pool1 = nn.MaxPool2d(2,2) 

    self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU()
        ) 

    self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) 
    
    self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        )
    
    self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        )

    self.aap = nn.AdaptiveAvgPool2d((1,1))
  
  def forward(self, x):
    x = self.convblock1(x)
    x = self.convblock2(x)
    x = self.convblock3(x)
    x = self.pool1(x)
    x = self.convblock4(x)
    x = self.convblock5(x)
    x = self.convblock6(x)
    x = self.convblock7(x)
    x = self.aap(x)
    x = x.view(-1,10)
    return F.log_softmax(x,dim=1)
