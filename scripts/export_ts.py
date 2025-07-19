import torch
from graphnn.engine.model import HandGNN

model = HandGNN()
model.load_state_dict(torch.load("outputs/checkpoints/best.pth", map_location="cpu"))
model.eval()
example = torch.randn(1,50,39)
ts = torch.jit.trace(model, example)
ts.save("outputs/teacher_pred.pt")
print("TorchScript saved to outputs/teacher_pred.pt")
