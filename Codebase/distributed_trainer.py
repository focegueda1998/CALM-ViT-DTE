import os
from time import time, sleep
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
from torchvision.transforms.functional import InterpolationMode
import sys

parent_dir = "/config"

def save_samples(batch, attention_maps, y_pred, y_actual):
    try:
        for i, attention_map in enumerate(attention_maps):
            sample = batch[i]
            sample = torchvision.transforms.Normalize(
                    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                    std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
            )(sample)
            sample = sample.permute(1, 2, 0).cpu().detach().numpy()
            Image.fromarray((sample * 255).astype(np.uint8)).save(f"{parent_dir}/Codebase/samples/{i}_sample.jpg")
            full_attention = attention_map
            f_min = full_attention.min()
            f_max = full_attention.max()
            f_diff = f_max - f_min if f_max - f_min > 1e-6 else 1e-6
            full_attention = (full_attention - f_min) / f_diff
            full_attention_2 = full_attention.mul(255).clamp(0, 255).reshape(14, 14).cpu().detach().numpy()
            Image.fromarray(full_attention_2.astype(np.uint8)).save(f"{parent_dir}/Codebase/samples/{i}_sample_attn.jpg")
            full_attention, _ =  torch.sort(full_attention)
            full_attention = full_attention.cpu().detach().numpy()
            plt.scatter(range(196), full_attention)
            plt.grid(True)
            plt.xlabel("Attention Heads")
            plt.ylabel("Attention Weights")
            plt.title(f"Predicted: {int(y_pred[i])}, Actual: {int(y_actual[i])}")
            plt.savefig(f"{parent_dir}/Codebase/samples/{i}_sample_attn_scatter.jpg")
            plt.close()
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

    gc.collect()
    if use_gpu: torch.cuda.empty_cache()
    os.environ['NCCL_DEBUG'] = 'ERROR'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
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
    # attention_maps = None
    # batch = None
    # y_pred = None
    # y_actual = None
    model.train()
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        # dataset.reshuffle()
        # dataset.train = True
        epoch_loss = 0.0
        predicted = 0
        model.train()
        for i, (x, y) in enumerate(dataloader):
            # batch = x
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_hat = model(x)
            y = y.float()
            # y_actual = y
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
        if global_rank == 0 and local_rank == 0 and ((epoch + 1) % 1 == 0):
            torch.save(model.module.state_dict(), f"{parent_dir}/Codebase/models/model_b.pth")
            print("Model saved to models/model_b.pth")
            # save_samples(batch, attention_maps, y_pred, y_actual)
    model = model.to("cpu")
    dist.destroy_process_group()
    return model.module

if __name__ == "__main__":
    start = time()
    call(['mkdir', '-p', f'{parent_dir}/Codebase/models'])
    call(['mkdir', '-p', f'{parent_dir}/Codebase/samples'])
    sc = SparkContext.getOrCreate()
    spark = SparkSession.builder.getOrCreate()
    distributor = TorchDistributor(num_processes=8, local_mode=False, use_gpu=True)
    model = rvh.initialize_vit(torch.device("cuda" if torch.cuda.is_available() else "cpu"), weights=f"", type="b")
    print(model)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224), antialias=True),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    opt = optim.Adam(model.parameters(), lr=0.00015625)
    crit = torch.nn.BCEWithLogitsLoss()
    dataset = rvh.ImageDataset(f"{parent_dir}/AI_Human_Generated_Images/", "train.csv", transform=transform, split_ratio=1, train=True)
    model = distributor.run(
        train,
        model,
        optimizer=opt,
        scheduler=None,
        criterion=crit,
        use_gpu=True,
        dataset=dataset,
        epochs=66,
        batch_size=96
    )
    torch.save(model.state_dict(), f"{parent_dir}/Codebase/models/model_b.pth")
    print(f"Model saved to {parent_dir}/Codebase/models/model_b.pth")
    # sleep(30)
    # save_samples(batch, attention_maps, y_pred, y_actual)
    print(f"Time taken: {time() - start}")
    sc.stop()