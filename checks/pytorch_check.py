import torch
from keras.src.ops import dtype
import time

def create_torch_tensors(device):
    x = torch.rand((10000, 10000), dtype = torch.float32)
    y = torch.rand((10000, 10000), dtype = torch.float32)

    return x, y

# with gpu
print("With GPU ......")
t1 = time.time()
print("start time " + str(t1))
device = torch.device("mps")
x, y = create_torch_tensors(device)
z = x@y
# Important: synchronize GPU before measuring time
torch.mps.synchronize()

t2 = time.time()
print("end time " +  str(t2))
print(
    "time taken to create tensors with gpu: } " +  str(t2 - t1)
)



t1 = time.time()
print("start time " + str(t1))
device = torch.device("cpu")
x, y = create_torch_tensors(device)
z= x@y
t2 = time.time()
print("end time " +  str(t2))
print(
    "time taken to create tensors with cpu: } " +  str(t2 - t1)
)



