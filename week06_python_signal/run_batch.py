from week06_python_signal.dataset import make_loaders
from week06_python_signal.model_cnn import ModulationCNN

train_loader, val_loader = make_loaders()
signals,labels = next(iter(train_loader))
logits = ModulationCNN()(signals)

print(signals.shape) #torch.Size([64,2,128])
print(labels.shape) #torch.Size([64])
print(logits.shape) #torch.Size([64,4])
