import os
from time import time, sleep
import torch
import random
import matplotlib.pyplot as plt
import numpy as np
import pyspark
import CALM_ViT_V2 as rvh
import gc
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from PIL import Image
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.torch.distributor import TorchDistributor
from torchvision.datasets import ImageNet
from subprocess import call
import torchvision
import torchvision.transforms.v2 as transforms
from torchvision.transforms.functional import InterpolationMode
import sys

parent_dir = "/config"

def train(initializer, optimizer, scheduler,
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
    from torch.utils.data import DataLoader, DistributedSampler, default_collate

    print("Training on device", os.environ['RANK'])

    gc.collect()
    if use_gpu: torch.cuda.empty_cache()
    os.environ['NCCL_DEBUG'] = 'ERROR'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
    dist.init_process_group(backend='nccl' if use_gpu else 'gloo')

    local_rank = int(os.environ['LOCAL_RANK'])
    global_rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    device = torch.device(f"cuda:{local_rank}" if use_gpu else "cpu")
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)
    model = initializer
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    sampler = DistributedSampler(dataset, shuffle=True, seed=2006)
    sampler.set_epoch(0)
    cut_mix = transforms.CutMix(num_classes=1000, alpha=1.0)
    mix_up = transforms.MixUp(num_classes=1000, alpha=0.8)
    mix_both = transforms.RandomChoice([cut_mix, mix_up])
    def collate_fn(batch): return mix_both(*default_collate(batch))
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn)
    criterion = torch.nn.CrossEntropyLoss()
    # criterion_mse = torch.nn.MSELoss()
    # criterion_kl = torch.nn.KLDivLoss(reduction='batchmean')
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
            # Compute classification and reconstruction loss on the batch
            y_hat = model(x)
            loss = criterion(y_hat.squeeze(), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            # Determine accuracy for the DOMINANT class because I don't want to write code for mixup/cutmix soft accuracy
            _, predicted = torch.max(y_hat.data, 1)
            _, y_labels = torch.max(y.data, 1)  
            correct = (predicted == y_labels).sum().item()
            accuracy = correct / y.size(0)
            print(f"Epoch: {epoch + 1}, Batch: {i + 1}, Device: [{global_rank}, {local_rank}], Loss: {loss}, Accuracy: {accuracy * 100:.4f}%")
        if global_rank == 0 and local_rank == 0:
            torch.save(model.module.state_dict(), f"{parent_dir}/Codebase/models/model_cls.pth")
            print("Model saved to models/model_cls.pth")
        if scheduler is not None:
            scheduler.step()
            if global_rank == 0 and local_rank == 0:
                print(f"Epoch {epoch + 1} scheduler step!")
    model = model.to("cpu")
    dist.destroy_process_group()
    return model.module

if __name__ == "__main__":
    start = time()
    call(['mkdir', '-p', f'{parent_dir}/Codebase/models'])
    call(['mkdir', '-p', f'{parent_dir}/Codebase/samples'])
    spark = SparkSession.builder.appName("CALM_ViT_Training").getOrCreate()
    distributor = TorchDistributor(num_processes=4, local_mode=False, use_gpu=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = rvh.ViT(device, type=8, heads=12, seq_length=224, in_features=672,
                 dim_step=48, mean_var_hidden=160,
                 seq_len_step=16, seq_len_reduce=96, out_features=1000,
                 force_reduce=False, generate=False)
    print(model)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model.load_state_dict(torch.load(f"{parent_dir}/Codebase/models/model_cls.pth", map_location=device, weights_only=True))
        print("Loaded existing model weights from model_cls.pth")
    except:
        print("No existing model weights found, starting fresh training.")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.ColorJitter(brightness=(0.5, 1), contrast=(0.5, 1), saturation=(0.5, 1), hue=(-0.125, 0.125)),
        transforms.RandomSolarize(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomGrayscale(),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    opt = optim.AdamW(model.parameters(), lr=3.1e-4, weight_decay=0.02)
    dataset = ImageNet(
        root="/dataset/imagenet/",
        split="train",
        transform=transform
    )
    model = distributor.run(
        train,
        model,
        optimizer=opt,
        scheduler=None,
        use_gpu=True,
        dataset=dataset,
        epochs=5,
        batch_size=320
    )
    torch.save(model.state_dict(), f"{parent_dir}/Codebase/models/model_cls.pth")
    print(f"Model saved to {parent_dir}/Codebase/models/model_cls.pth")
    print(f"Time taken: {time() - start}")
    sleep(30)
    spark.stop()