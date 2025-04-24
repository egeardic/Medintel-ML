import torch
print(torch.cuda.is_available())  # True yazmalı
print(torch.cuda.get_device_name(0))  # "NVIDIA GeForce GTX 1650 Ti" yazmalı
