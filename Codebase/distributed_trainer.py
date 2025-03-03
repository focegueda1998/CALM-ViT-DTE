import os
import torch
import random
import matplotlib.pyplot as plt
import numpy as np
import pyspark
import reverse_ViT_hybrid as rvh
import gc
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from PIL import Image
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.torch.distributor import TorchDistributor
from subprocess import call
import torchvision

def train(initializer, optimizer, scheduler, criterion,
          use_gpu=True, dataset=None, epochs=15, batch_size=128):
    import torch
    import os
    import gc
    import torch.optim as optim
    from torch.optim.lr_scheduler import StepLR
    import torch.nn as nn
    import time
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader, DistributedSampler

    print("Training on device", os.environ['RANK'])

    os.environ["NCCL_SOCKET_IFNAME"] = "eth0"
    gc.collect()
    if use_gpu: torch.cuda.empty_cache()
    os.environ['NCCL_DEBUG'] = 'ERROR'
    dist.init_process_group(backend='nccl' if use_gpu else 'gloo')

    local_rank = int(os.environ['LOCAL_RANK'])
    global_rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    device = torch.device(f"cuda:{local_rank}" if use_gpu else "cpu")
    scheduler = StepLR(optimizer, step_size=4, gamma=0.5)
    model = initializer
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    sampler = DistributedSampler(dataset, shuffle=True)
    sampler.set_epoch(0)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, pin_memory=True)
    attention_map = None
    model.train()
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        # dataset.reshuffle()
        # dataset.train = True
        epoch_loss = 0.0
        predicted = 0
        model.train()
        for i, (x, y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_hat, attention_map = model(x)
            y = y.float()
            loss = criterion(y_hat.squeeze(), y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            y_prob = torch.sigmoid(y_hat)
            y_pred = (y_prob > 0.5).float().squeeze()
            predicted = (y_pred == y).float().sum()
            print(f"Epoch: {epoch + 1}, Batch: {i + 1}, Device: {global_rank}, Loss: {loss}, Predicted: {int(predicted)}/{len(y_pred)}")
        if scheduler is not None:
            scheduler.step()
        if global_rank == 0 and ((epoch + 1) % 4 == 0 or epoch + 1 == epochs or epoch == 0):
            torch.save(model.module.state_dict(), "/config/Codebase/models/model.pth")
            print("Model saved to /config/Codebase/models/model.pth")
    model = model.to("cpu")
    dist.destroy_process_group()
    return model.module, attention_map

if __name__ == "__main__":
    call(['mkdir', '-p', '/config/Codebase/models'])
    sc = SparkContext.getOrCreate()
    spark = SparkSession.builder.getOrCreate()
    distributor = TorchDistributor(num_processes=4, local_mode=False, use_gpu=True)
    model = rvh.initialize_vit(torch.device("cuda" if torch.cuda.is_available() else "cpu"), weights="DEFAULT")
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x)
    ])
    opt = optim.Adam(model.parameters(), lr=0.001)
    sched = StepLR(opt, step_size=4, gamma=0.5)
    crit = torch.nn.BCEWithLogitsLoss()
    dataset = rvh.ImageDataset("/config/AI_Human_Generated_Images/", "train.csv", transform=transform, split_ratio=1, train=True)
    model, attention_map = distributor.run(
        train,
        model,
        optimizer=opt,
        scheduler=None,
        criterion=crit,
        use_gpu=True,
        dataset=dataset,
        epochs=16,
        batch_size=128
    )
    torch.save(model.state_dict(), "/config/Codebase/models/model.pth")
    print("Model saved to /config/Codebase/models/model.pth")
    print(f"Attention map shape: {attention_map.shape}")
    print(f"Attention map type: {attention_map.dtype}")
    try:
        full_attention = attention_map[0]  # First sample, full attention matrix
        full_attention = torch.nn.functional.normalize(full_attention, p=1, dim=1)
        full_attention = full_attention.mul(255).clamp(0, 255).byte()
        plt.figure(figsize=(10, 10))
        plt.imshow(full_attention.cpu().detach().numpy(), cmap='viridis')
        plt.colorbar()
        plt.title('Full Attention Matrix')
        plt.savefig("full_attention_matrix.jpg")
        plt.close()
    except Exception as e:
        print(f"Something went wrong: {e}")
    sc.stop()