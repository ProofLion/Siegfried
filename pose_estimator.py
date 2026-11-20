"""Estimate a pose [x, y, z, qw, qx, qy, qz] from an image with a CNN."""

# Sources: NeuralNets_Participation_26Fa Notebook from class,
#          https://www.geeksforgeeks.org/python/working-images-python/,
#          https://www.geeksforgeeks.org/python/how-to-convert-images-to-numpy-array/,
#          https://docs.pytorch.org/docs/2.15/nn.html,
#          https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from data_loader import load_trajectory

MODEL_PATH = Path(__file__).resolve().parent / "pose_model.pt"
POSE_DIMS = 7  # x, y, z, qw, qx, qy, qz
MAX_PIXEL_VALUE = 255.0

loaded_models = {}


def load_image(image_path, image_size=(120, 192)):
    """Return one image as a tensor of shape (1, 1, height, width), values 0 to 1."""
    height, width = image_size
    image = Image.open(image_path).resize((width, height))
    number_grid = (
        np.asarray(image, dtype=np.float32) / MAX_PIXEL_VALUE
    )  # shape (height, width)
    return torch.tensor(number_grid).reshape(
        1, 1, height, width
    )  # (batch, channels, height, width)


def load_dataset(
    frame_step=10,
    data_dir="shirtv1",
    trajectory="roe1",
    domain="synthetic",
    image_size=(120, 192),
):
    """Load evenly spaced frames and return images (N, 1, height, width) and poses (N, POSE_DIMS)."""
    frames, _ = load_trajectory(data_dir, trajectory, domain)
    frames = frames[::frame_step]
    x = torch.cat([load_image(f["image_path"], image_size) for f in frames])
    y = torch.tensor([f["pose"] for f in frames], dtype=torch.float32)
    return x, y


def split_fold(x, y, fold, k=20):
    """Hold out block number fold of k consecutive blocks and return x_train, y_train, x_val, y_val."""
    n = len(x)
    start = n * fold // k
    end = n * (fold + 1) // k

    x_val, y_val = x[start:end], y[start:end]

    x_train = torch.cat([x[:start], x[end:]])
    y_train = torch.cat([y[:start], y[end:]])

    return x_train, y_train, x_val, y_val


def build_model(
    num_filters=16,
    filter_growth=2,
    num_blocks=3,
    kernel_size=3,
    pool_size=(4, 6),
    activation_func=nn.ReLU,
):
    """Build a CNN that maps one grayscale image to a pose (x, y, z, qw, qx, qy, qz)."""
    layers = []
    in_channels = 1
    out_channels = num_filters

    for _ in range(num_blocks):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=1))
        layers.append(activation_func())
        layers.append(nn.MaxPool2d(2))  # halve height and width
        in_channels = out_channels
        out_channels = out_channels * filter_growth

    layers.append(
        nn.AdaptiveAvgPool2d(pool_size)
    )  # average each feature map down to a pool size grid
    layers.append(nn.Flatten())  # flatten for the linear layer
    layers.append(
        nn.Linear(in_channels * pool_size[0] * pool_size[1], POSE_DIMS)
    )  # calculate pose

    return nn.Sequential(*layers)


def position_loss(guess_pose, true_pose, loss_func=F.mse_loss):
    """Loss on x, y, z (the first 3 numbers of the pose)."""
    return loss_func(guess_pose[:, :3], true_pose[:, :3])


def rotation_loss(guess_pose, true_pose, loss_func=F.mse_loss):
    """Loss on the quaternion (the last 4 numbers of the pose)."""
    guess_q = F.normalize(guess_pose[:, 3:], dim=1)  # scale to length 1
    true_q = true_pose[:, 3:]

    dot = (guess_q * true_q).sum(dim=1, keepdim=True)  # negative = opposite sign
    guess_q = guess_q * torch.where(dot >= 0, 1.0, -1.0)  # flip to match

    return loss_func(guess_q, true_q)


def total_pose_loss(
    guess_pose,
    true_pose,
    position_loss_func=F.mse_loss,
    rotation_loss_func=F.mse_loss,
    rotation_weight=1.0,
):
    """Return position loss + rotation loss for poses of shape (batch, POSE_DIMS)."""
    position = position_loss(guess_pose, true_pose, position_loss_func)
    rotation = rotation_loss(guess_pose, true_pose, rotation_loss_func)
    return position + rotation_weight * rotation


def train_model(
    model,
    x_train,
    y_train,
    epochs=100,
    batch_size=16,
    learning_rate=0.01,
    optimizer_class=torch.optim.Adam,
    rotation_weight=1.0,
):
    """Train the model in place and print the average loss each epoch."""
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    optimizer = optimizer_class(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        total_loss = 0.0
        for images, true_poses in train_loader:
            prediction = model(images)  # forward pass
            loss = total_pose_loss(
                prediction, true_poses, rotation_weight=rotation_weight
            )  # compute loss

            optimizer.zero_grad()  # clear old gradients
            loss.backward()  # backward pass
            optimizer.step()  # update the weights

            total_loss += loss.item() * len(images)
        print(f"Epoch {epoch + 1}/{epochs} - loss: {total_loss / len(x_train):.4f}")


def train_and_save(
    x,
    y,
    path=MODEL_PATH,
    epochs=100,
    learning_rate=0.01,
    rotation_weight=1.0,
    pool_size=(4, 6),
    image_size=(120, 192),
    seed=0,
    num_blocks=3,
):
    """Train one model on all the frames and save it for estimate_pose."""
    pos_mean = y[:, :3].mean(dim=0)
    pos_std = y[:, :3].std(dim=0)
    y_scaled = y.clone()
    y_scaled[:, :3] = (y[:, :3] - pos_mean) / pos_std

    torch.manual_seed(seed)
    model = build_model(pool_size=pool_size, num_blocks=num_blocks)
    train_model(
        model,
        x,
        y_scaled,
        epochs=epochs,
        learning_rate=learning_rate,
        rotation_weight=rotation_weight,
    )

    torch.save(
        {
            "weights": model.state_dict(),
            "model_settings": {"pool_size": pool_size, "num_blocks": num_blocks},
            "image_size": image_size,
            "pos_mean": pos_mean,
            "pos_std": pos_std,
        },
        path,
    )
    return model, pos_mean, pos_std


def position_error_m(guess_pose, true_pose):
    """Distance in meters between guessed and true positions."""
    diff = guess_pose[:, :3] - true_pose[:, :3]
    return torch.linalg.norm(diff, dim=1)


def rotation_error_deg(guess_pose, true_pose):
    """Angle in degrees between guessed and true rotations, one number per image."""
    guess_q = F.normalize(guess_pose[:, 3:], dim=1)  # scale to length 1
    true_q = true_pose[:, 3:]

    dot = (guess_q * true_q).sum(dim=1)
    dot = dot.abs()  # q and -q are the same rotation
    dot = dot.clamp(max=1.0)  # rounding can push it just over 1

    angle_rad = 2 * torch.acos(dot)
    return torch.rad2deg(angle_rad)


def predict_poses(model, images, pos_mean, pos_std, batch_size=16):
    """Run the model on images and return poses with positions converted back to meters."""
    model.eval()
    with torch.inference_mode():
        guesses = []
        for start in range(0, images.size(0), batch_size):
            batch = images[start : start + batch_size]  # next batch_size images
            guesses.append(model(batch))
        guess = torch.cat(guesses)

        position = guess[:, :3] * pos_std + pos_mean  # undo the standardizing
        quaternion = guess[:, 3:]
        return torch.cat([position, quaternion], dim=1)


def run_fold(
    fold,
    x1,
    y1,
    x2,
    y2,
    k=20,
    epochs=100,
    learning_rate=0.01,
    rotation_weight=1.0,
    pool_size=(4, 6),
    seed=0,
    num_blocks=3,
):
    """Train on all but one block of each trajectory and return errors on the validation blocks."""
    x1_train, y1_train, x1_val, y1_val = split_fold(x1, y1, fold, k)
    x2_train, y2_train, x2_val, y2_val = split_fold(x2, y2, fold, k)

    x_train = torch.cat([x1_train, x2_train])
    y_train = torch.cat([y1_train, y2_train])
    x_val = torch.cat([x1_val, x2_val])
    y_val = torch.cat([y1_val, y2_val])

    pos_mean = y_train[:, :3].mean(dim=0)
    pos_std = y_train[:, :3].std(dim=0)
    y_train_scaled = y_train.clone()
    y_train_scaled[:, :3] = (y_train[:, :3] - pos_mean) / pos_std

    torch.manual_seed(seed)
    model = build_model(
        pool_size=pool_size, num_blocks=num_blocks
    )  # fresh model every fold
    train_model(
        model,
        x_train,
        y_train_scaled,
        epochs=epochs,
        learning_rate=learning_rate,
        rotation_weight=rotation_weight,
    )

    guess = predict_poses(model, x_val, pos_mean, pos_std)
    baseline_pose = y_train.mean(dim=0, keepdim=True).repeat(len(y_val), 1)

    return [
        position_error_m(guess, y_val).mean().item(),
        rotation_error_deg(guess, y_val).mean().item(),
        position_error_m(baseline_pose, y_val).mean().item(),
        rotation_error_deg(baseline_pose, y_val).mean().item(),
    ]


def cross_validate(x1, y1, x2, y2, k=20, **fold_settings):
    """Run every fold, print the per-fold and average errors, and return the averages."""
    results = []
    for fold in range(k):
        print(f"--- Fold {fold + 1}/{k} ---")
        results.append(run_fold(fold, x1, y1, x2, y2, k, **fold_settings))

    results = torch.tensor(results)  # shape (k, 4)
    print("per fold [model pos m, model rot deg, baseline pos m, baseline rot deg]:")
    print(results)

    average = results.mean(dim=0)
    print("model position error (m):", average[0].item())
    print("model rotation error (deg):", average[1].item())
    print("baseline position error (m):", average[2].item())
    print("baseline rotation error (deg):", average[3].item())
    return average


def load_saved_model(model_path):
    """Return the trained model, plus the saved image size and position mean and standard deviation."""
    saved = torch.load(model_path, weights_only=True)
    model = build_model(**saved["model_settings"])
    model.load_state_dict(saved["weights"])
    model.eval()
    return model, saved


def estimate_pose(image_path, model_path=MODEL_PATH):
    """Return [x, y, z, qw, qx, qy, qz] in meters with a unit quaternion."""
    if model_path not in loaded_models:
        loaded_models[model_path] = load_saved_model(model_path)
    model, saved = loaded_models[model_path]

    image = load_image(image_path, saved["image_size"])  # shape (1, 1, height, width)
    pose = predict_poses(model, image, saved["pos_mean"], saved["pos_std"])[
        0
    ]  # [0] = the only image

    quaternion = F.normalize(pose[3:], dim=0)  # scale to length 1
    if quaternion[0] < 0:  # q and -q are the same rotation, so keep w positive
        quaternion = -quaternion
    return torch.cat([pose[:3], quaternion]).numpy().astype(np.float64)


if __name__ == "__main__":
    x1, y1 = load_dataset(trajectory="roe1")
    x2, y2 = load_dataset(trajectory="roe2")

    # cross_validate(x1, y1, x2, y2) # measure the model (20 folds)
    train_and_save(
        torch.cat([x1, x2]), torch.cat([y1, y2])
    )  # train and save the final model
