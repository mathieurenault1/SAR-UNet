import torch
from torch import nn
import numpy as np
import os
import pickle
from tqdm import tqdm
from cloud_cover import cloud_cover_dataset 
from models import models

def get_model_class(model_file):
    if "SmaAt_UNet_cloud" in model_file:
        model_name = "SmaAt_UNet_cloud"
        model = models.SmaAt_UNet_cloud
    elif "SAR_UNet_cloud" in model_file:
        model_name = "SAR_UNet_cloud"
        model = models.SAR_UNet_cloud
    elif "UNet" in model_file:
        model_name = "UNet"
        model = models.UNet
    else:
        raise NotImplementedError(f"Model not found")
    return model, model_name


def compute_model_metrics(model, test_dl, loss="mse"):
    model.eval()  # or model.freeze()?
    model.to("cuda")
    if loss.lower() == "mse":
        loss_func = nn.functional.mse_loss
    elif loss.lower() == "mae":
        loss_func = nn.functional.l1_loss
    with torch.no_grad():
        threshold = 0.5
        total_tp = 0
        total_fp = 0
        total_tn = 0
        total_fn = 0
        loss_model = 0.0
        for x, y_true in tqdm(test_dl, leave=False):
            x = x.to("cuda")
            y_true = y_true.to("cuda")
            y_pred = model(x)
            loss_model += loss_func(y_pred.squeeze(), y_true)
            # denormalize and convert from mm/5min to mm/h
            y_pred_adj = y_pred.squeeze()
            y_true_adj = y_true.squeeze()
            # convert to masks for comparison
            y_pred_mask = y_pred_adj > threshold
            y_true_mask = y_true_adj > threshold

            tn, fp, fn, tp = np.bincount(y_true_mask.cpu().view(-1) * 2 + y_pred_mask.cpu().view(-1), minlength=4)
            total_tp += tp
            total_fp += fp
            total_tn += tn
            total_fn += fn
            # get metrics for sample
            precision = total_tp / (total_tp + total_fp)
            recall = total_tp / (total_tp + total_fn)
            accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn)
            f1 = 2 * precision * recall / (precision + recall)
        loss_model /= len(test_dl)
    return loss_model, precision, recall, accuracy, f1

def compute_persistence_metrics(test_dl, loss="mse"):
    if loss.lower() == "mse":
        loss_func = nn.functional.mse_loss
    elif loss.lower() == "mae":
        loss_func = nn.functional.l1_loss
    threshold = 0.5
    total_tp = 0
    total_fp = 0
    total_tn = 0
    total_fn = 0
    loss_model = 0.0
    for x, y_true in tqdm(test_dl, leave=False):
        y_pred = x[:, -1, :]
        y_pred= y_pred.expand(6, 256, 256)
        loss_model += loss_func(y_pred.squeeze(), y_true)
        # denormalize and convert from mm/5min to mm/h
        y_pred_adj = y_pred.squeeze()
        y_true_adj = y_true.squeeze()
        # convert to masks for comparison
        y_pred_mask = y_pred_adj > threshold
        y_true_mask = y_true_adj > threshold

        tn, fp, fn, tp = np.bincount(y_true_mask.view(-1) * 2 + y_pred_mask.view(-1), minlength=4)
        total_tp += tp
        total_fp += fp
        total_tn += tn
        total_fn += fn
        # get metrics for sample
        precision = total_tp / (total_tp + total_fp)
        recall = total_tp / (total_tp + total_fn)
        accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn)
        f1 = 2 * precision * recall / (precision + recall)
    loss_model /= len(test_dl)
    return loss_model, precision, recall, accuracy, f1


def get_persistent_metrics(data_file, loss="mse"):
    dataset = cloud_cover_dataset.cloud_maps(
        folder=data_file,
        in_channels=4,
        out_channels=6, train=False)

    test_dl = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    loss_persistence, precision, recall, accuracy, f1 = compute_persistence_metrics(test_dl, loss=loss)
    print(f"Loss Persistence (MSE): {loss_persistence}, precision: {precision}, recall: {recall}, accuracy: {accuracy}, f1: {f1}")
    return loss_persistence, precision, recall, accuracy, f1


def get_all_metrics(model_folder, data_file,  loss="mse"):
    # Save it to a dict that can be saved (and plotted)
    test_metrics = dict()
    persistence_metrics = get_persistent_metrics(data_file)
    test_metrics["Persistence"] = persistence_metrics

    models = [m for m in os.listdir(model_folder) if ".ckpt" in m]
    dataset = cloud_cover_dataset.cloud_maps(
        folder=data_file,
        in_channels=4,
        out_channels=6, train=False)

    test_dl = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # load the models
    for model_file in tqdm(models, desc="Models", leave=True):
        model, model_name = get_model_class(model_file)
        model = model.load_from_checkpoint(f"{model_folder}/{model_file}")
        model_metrics = compute_model_metrics(model, test_dl, loss)

        test_metrics[model_name] = model_metrics
    return test_metrics

if __name__ == '__main__':
    loss = "mse"
    # Models that are compared should be in this folder (the ones with the lowest validation error)
    model_folder = "checkpoints/eval/cloud cover"
    data_file = 'data/cloud cover dataset'

    # This changes whether to load or to run the model loss calculation
    load = False
    if load:
        # load the losses
        with open(f"checkpoints/comparison/cloud cover/model_losses_{loss.upper()}.pkl", "rb") as f:
            test_metrics = pickle.load(f)

    else:
        test_metrics = get_all_metrics(model_folder, data_file, loss)
        # Save losses
        with open(model_folder + f"/model_losses_{loss.upper()}.pkl",
                  "wb") as f:
            pickle.dump(test_metrics, f)

    print(test_metrics)
