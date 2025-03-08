import os
from time import time
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
import sys

parent_dir = "/config"

def save_samples(batch, attention_maps):
    try:
        for i, attention_map in enumerate(attention_maps):
            sample = batch[i].permute(1, 2, 0).cpu().detach().numpy()
            Image.fromarray((sample * 255).astype(np.uint8)).save(f"{parent_dir}/Codebase/samples/{i}_sample.jpg")
            full_attention = attention_map.mul(255).clamp(0, 255).reshape(14, 14).cpu().detach().numpy()
            Image.fromarray(full_attention.astype(np.uint8)).save(f"{parent_dir}/Codebase/samples/{i}_sample_attn.jpg")
    except Exception as e:
        print(f"Something went wrong: {e}")

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
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
    model = initializer
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    sampler = DistributedSampler(dataset, shuffle=True)
    sampler.set_epoch(0)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, pin_memory=True)
    attention_maps = None
    batch = None
    model.train()
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        # dataset.reshuffle()
        # dataset.train = True
        epoch_loss = 0.0
        predicted = 0
        model.train()
        for i, (x, y) in enumerate(dataloader):
            batch = x
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_hat, attention_maps = model(x)
            y = y.float()
            loss = criterion(y_hat.squeeze(), y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            y_prob = torch.sigmoid(y_hat)
            y_pred = (y_prob > 0.5).float().squeeze()
            predicted = (y_pred == y).float().sum()
            print(f"Epoch: {epoch + 1}, Batch: {i + 1}, Device: [{global_rank}, {local_rank}], Loss: {loss}, Predicted: {int(predicted)}/{len(y_pred)}")
        if scheduler is not None:
            scheduler.step()
        if global_rank == 0 and local_rank == 0 and ((epoch + 1) % 10 == 0):
            torch.save(model.module.state_dict(), f"{parent_dir}/Codebase/models/model.pth")
            print("Model saved to models/model.pth")
            save_samples(batch, attention_maps)
    model = model.to("cpu")
    dist.destroy_process_group()
    return model.module, attention_maps, batch

if __name__ == "__main__":
    start = time()
    call(['mkdir', '-p', f'{parent_dir}/Codebase/models'])
    call(['mkdir', '-p', f'{parent_dir}/Codebase/samples'])
    sc = SparkContext.getOrCreate()
    spark = SparkSession.builder.getOrCreate()
    distributor = TorchDistributor(num_processes=8, local_mode=False, use_gpu=True)
    model = rvh.initialize_vit(torch.device("cuda" if torch.cuda.is_available() else "cpu"), weights=f"DEFAULT")
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x)
    ])
    opt = optim.Adam(model.parameters(), lr=0.1)
    crit = torch.nn.BCEWithLogitsLoss()
    dataset = rvh.ImageDataset(f"{parent_dir}/AI_Human_Generated_Images/", "train.csv", transform=transform, split_ratio=1, train=True)
    model, attention_maps, batch = distributor.run(
        train,
        model,
        optimizer=opt,
        scheduler=None,
        criterion=crit,
        use_gpu=True,
        dataset=dataset,
        epochs=75,
        batch_size=112
    )
    torch.save(model.state_dict(), f"{parent_dir}/Codebase/models/model.pth")
    print(f"Model saved to {parent_dir}/Codebase/models/model.pth")
    save_samples(batch, attention_maps)
    print(f"Time taken: {time() - start}")
    sc.stop()