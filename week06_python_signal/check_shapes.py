import torch
from week06_python_signal.model_cnn_Conv1d import ModulationCNN

model = ModulationCNN()
model.eval()

x = torch.randn(64,2,128)
print("input:   ", tuple(x.shape))

with torch.no_grad():
    for name, layer in model.features.named_children():
        x = layer(x)
        print(f"{name:10}", tuple(x.shape))

    x = x.squeeze(-1)
    print("squeeze: ",tuple(x.shape))

    logits = model.classifier(x)
    print("output:  ", tuple(logits.shape))