import os
import torch
import random
import pyspark
import reverse_ViT_hybrid as rvh
import gc
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.torch.distributor import TorchDistributor
import torchvision

def train(initializer, use_gpu=True, dataset=None, epochs=15, batch_size=128):
    import torch
    import os
    import gc
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
    model, optimizer, _, criterion = initializer
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    sampler = DistributedSampler(dataset, shuffle=True)
    sampler.set_epoch(0)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, pin_memory=True)
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
            y_hat, _ = model(x)
            y = y.float()
            loss = criterion(y_hat.squeeze(), y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            y_prob = torch.sigmoid(y_hat)
            y_pred = (y_prob > 0.5).float()
            y_pred = y_pred.squeeze().tolist()
            predicted = 0
            for j, pred in enumerate(y_pred):
                if pred == y[j]:
                    predicted += 1
            print(f"Epoch: {epoch}, Batch: {i}, Device: {global_rank}, Loss: {loss}, Predicted: {predicted}/{len(y_pred)}")

    model = model.to("cpu")
    dist.destroy_process_group()
    return model.module

if __name__ == "__main__":
    sc = SparkContext.getOrCreate()
    spark = SparkSession.builder.getOrCreate()
    distributor = TorchDistributor(num_processes=4, local_mode=False, use_gpu=True)
    model = rvh.initialize_vit(torch.device("cuda" if torch.cuda.is_available() else "cpu"), weights="/config/model.pth")
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x)
    ])
    dataset = rvh.ImageDataset("/config/AI_Human_Generated_Images/", "train.csv", transform=transform, split_ratio=1, train=True)
    model = distributor.run(
        train,
        model,
        use_gpu=True,
        dataset=dataset,
        epochs=1,
        batch_size=256
    )
    torch.save(model.state_dict(), "/config/model.pth")
    print("Model saved to /config/model.pth")
    sc.stop()