import torch
from torch import nn
import numpy as np
import os
import pickle
from tqdm import tqdm
from utils import dataset_precip
from models import models


def get_model_class(model_file):
    if "SmaAt_UNet_precip" in model_file:
        model_name = "SmaAt_UNet_precip"
        model = models.SmaAt_UNet_precip
    elif "SAR_UNet_precip" in model_file:
        model_name = "SAR_UNet_precip"
        model = models.SAR_UNet_precip
    elif "UNet" in model_file:
        model_name = "UNet_precip"
        model = models.UNet_precip
    else:
        raise NotImplementedError(f"Model not found")
    return model, model_name


def compute_model_metrics(model, test_dl, denormalize=True):
    model.eval()  # or model.freeze()?
    model.to(device)
    loss_func = nn.functional.mse_loss
    factor = 1
    if denormalize:
        factor = 47.83
    with torch.no_grad():
        threshold = 0.5
        total_tp = 0
        total_fp = 0
        total_tn = 0
        total_fn = 0
        loss_model = 0.0
        for x, y_true in tqdm(test_dl, leave=False):
            x = x.to(device)
            y_true = y_true.to(device)
            y_pred = model(x)
            loss_model += loss_func(y_pred.squeeze() * factor, y_true * factor) / y_true.size(0)
            # denormalize and convert from mm/5min to mm/h
            y_pred_adj = y_pred.squeeze() * 47.83 * 12
            y_true_adj = y_true.squeeze() * 47.83 * 12
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

def compute_persistence_metrics(test_dl,denormalize=True):
    loss_func = nn.functional.mse_loss
    factor = 1
    if denormalize:
        factor = 47.83
    threshold = 0.5
    total_tp = 0
    total_fp = 0
    total_tn = 0
    total_fn = 0
    loss_model = 0.0
    for x, y_true in tqdm(test_dl, leave=False):
        y_pred = x[:, -1, :]
        loss_model += loss_func(y_pred.squeeze() * factor, y_true * factor) / y_true.size(0)
        # denormalize and convert from mm/5min to mm/h
        y_pred_adj = y_pred.squeeze() * 47.83 * 12
        y_true_adj = y_true.squeeze() * 47.83 * 12
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


def get_persistent_metrics(data_file, loss="mse", denormalize=True, in_channels=12):
    dataset = dataset_precip.precipitation_maps_oversampled_h5(
        folder=data_file,
        in_channels=in_channels,
        train=False)

    test_dl = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    loss_persistence, precision, recall, accuracy, f1 = compute_persistence_metrics(test_dl,denormalize=denormalize)
    print(f"Loss Persistence (MSE): {loss_persistence}, precision: {precision}, recall: {recall}, accuracy: {accuracy}, f1: {f1}")
    return loss_persistence, precision, recall, accuracy, f1


def get_all_metrics(model_folder, data_file, denormalize=True, in_channels=12):
    # Save it to a dict that can be saved (and plotted)
    test_metrics = dict()
    #persistence_metrics = get_persistent_metrics(data_file, in_channels)
    #test_metrics["Persistence"] = persistence_metrics

    models = [m for m in os.listdir(model_folder) if ".ckpt" in m]
    dataset = dataset_precip.precipitation_maps_oversampled_h5(
        folder=data_file,
        in_channels=in_channels,
        train=False)

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
        model_metrics = compute_model_metrics(model, test_dl, denormalize=denormalize)

        test_metrics[model_name] = model_metrics
    return test_metrics

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    denormalize = True
    in_channels = 12
    # Models that are compared should be in this folder (the ones with the lowest validation error)
    model_folder = "checkpoints/eval/precip"
    data_file = 'data/precip/train_test_2016-2019_input-length_12_img-ahead_6_rain-threshold_50.h5'

    # This changes whether to load or to run the model loss calculation
    load = False
    if load:
        # load the losses
        with open(f"checkpoints/eval/precip/model_losses_denormalized.pkl", "rb") as f:
            test_metrics = pickle.load(f)

    else:
        test_metrics = get_all_metrics(model_folder, data_file, denormalize, in_channels)
        # Save losses
        with open(model_folder + f"/model_losses_{f'de' if denormalize else ''}normalized.pkl",
                  "wb") as f:
            pickle.dump(test_metrics, f)

    print(test_metrics)


