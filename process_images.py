from PIL import Image
from typing import List, Iterable
from torchvision.transforms import functional as F
from torch import Tensor
from matplotlib import pyplot as plt
from pathlib import Path
import torch
from hopfield import SparseHopfield

def image_chunking(img: Image.Image, chunk_length: int, chunks_per_row: int, num_rows: int) -> Tensor:
    height = num_rows
    width = chunk_length * chunks_per_row

    img = img.resize((width, height))
    img = img.convert('L')
    tensor = F.to_tensor(img)
    return tensor.squeeze()

def process_images(imgs: List[Image.Image], chunk_length: int, chunks_per_row: int, num_rows: int) -> Tensor:
    tensors = [image_chunking(img, chunk_length, chunks_per_row, num_rows) for img in imgs]
    return torch.stack(tensors)

def images_in_path(path: Path | str, extensions: Iterable[str] = ('.jpg', '.png')) -> List[Image.Image]:
    if isinstance(path, str):
        path = Path(path)
    return [Image.open(p) for p in path.glob('*') if p.suffix in extensions]

def prepare_for_network(images: Tensor, chunks_per_row: int, num_rows: int) -> Tensor:
    batch_size, height, width = images.shape
    processed = images.reshape(batch_size, chunks_per_row * num_rows, -1)
    _, receptive_fields, field_dim = processed.shape
    return processed, receptive_fields, field_dim, height, width

def visualizable(image: Tensor, height: int, width: int) -> Tensor:
    return image.reshape(-1, height, width)

def prediction(input_image: Tensor, net: SparseHopfield, height: int, width: int) -> Tensor:
    assert input_image.dim() == 2 # currentl cannot handle batches
    prediction = net.predict(input_image.unsqueeze(0))
    return visualizable(prediction, height, width)[0]

def corruption_replace(data, noise_level=0.1):
    # corrupts data based on noise level, each value has a `noise_level` chance of being replaced
    mask = torch.rand_like(data) < noise_level
    noise = torch.randn_like(data)
    return torch.where(mask, noise, data)

def corruption_additive(data, noise_level=0.1):
    # corrupts data based on noise level, each value has a `noise_level` chance of being added to it
    mask = torch.rand_like(data) < noise_level
    noise = torch.randn_like(data)
    return data + torch.where(mask, noise, torch.zeros_like(data))

def corruption_revolving_lantern(data, noise_level=0.1, right_to_left=False):
    # corrupts data based on noise level. Starting from either the right or left, it will
    # replace the first `noise_level` percentage of the data with noise and then return
    # the data. If `right_to_left` is True, it will start from the right, otherwise the left
    batch, rows, cols = data.shape
    new_data = data.clone()
    noise_cols = int(cols * noise_level)
    noise_cols = min(noise_cols, cols)
    noise = torch.randn(batch, rows, noise_cols)
    if right_to_left:
        new_data[:, :, -noise_cols:] = noise
    else:
        new_data[:, :, :noise_cols] = noise
    return new_data

def blackouts(data, noise_level=0.1):
    # corrupts data based on noise level. It will randomly blackout a percentage of the data
    # based on the noise level
    mask = torch.rand_like(data) < noise_level
    return torch.where(mask, torch.zeros_like(data), data)

def blackout_revolving_lantern(data, noise_level=0.1, right_to_left=False):
    # corrupts data based on noise level. Starting from either the right or left, it will
    # replace the first `noise_level` percentage of the data with zeros and then return
    # the data. If `right_to_left` is True, it will start from the right, otherwise the left
    batch, rows, cols = data.shape
    new_data = data.clone()
    noise_cols = int(cols * noise_level)
    noise_cols = min(noise_cols, cols)
    if right_to_left:
        new_data[:, :, -noise_cols:] = torch.zeros_like(new_data[:, :, -noise_cols:])
    else:
        new_data[:, :, :noise_cols] = torch.zeros_like(new_data[:, :, :noise_cols])
    return new_data

def plot_tensors(tensor_dict):
    # given a dictionary of tensors, plot them, the key will be used as the titles
    num_tensors = len(tensor_dict)
    fig, axes = plt.subplots(1, num_tensors, figsize=(6 * num_tensors, 6))

    if num_tensors == 1:
        axes = [axes]

    for ax, (title, tensor) in zip(axes, tensor_dict.items()):
        ax.imshow(tensor.squeeze(0), cmap='gray')
        ax.set_title(title)

    plt.show()