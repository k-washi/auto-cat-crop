import sys
import os
import pathlib
path = pathlib.Path(__file__)
sys.path.append(str(path.resolve()))

import torch
import torch.nn as nn
from torchsummary import summary
import torch.optim as optim
from tqdm import tqdm
import datetime

import optuna
optuna.logging.disable_default_handler()

from config.config import cnf
from data.pet_dataloader import create_dataloader
from model.optuna_model import Net
from modules.mlflow_writer import MlflowWriter
from modules.early_stoping import EarlyStopping
from modules.logger import get_logger
logger = get_logger(__file__)


def train():
    logger.info("start train")
    os.environ["MLFLOW_TRACKING_URI"] = cnf.MLFLOW_TRACKING_URI
    study = optuna.create_study()
    study.optimize(_objective, n_trials=cnf.train.trial_size)
    logger.info(study.best_params)

    writer = MlflowWriter(cnf.experiment_name)
    writer.log_params_from_omegaconf_dict(study.best_params)

    for key, value in study.best_params:
        writer.log_param(key, value)
    writer.log_param("best_study_value", study.best_value)


def _objective(trial):
    logger.info("start trial")
    # 畳み込み層の数
    num_layer = trial.suggest_int('num_layer', 2, 4)

    # FC層のユニット数
    mid_units = [int(trial.suggest_discrete_uniform(
        "mid_units", 100, 500, 100)) for _ in range(2)]  # 2層に設定されている。

    # 各畳込み層のフィルタ数
    num_filters = [int(trial.suggest_discrete_uniform(
        "num_filter_" + str(i), 16, 128, 16)) for i in range(num_layer)]

    cnf.model.conv_num = num_layer
    cnf.model.mid_units = mid_units
    cnf.model.num_filters = num_filters

    now = datetime.datetime.now()
    nowf = now.strftime("%Y%m%d-%H%M%S")
    cnf.model_weight_prefix = "pet_cls_" + str(nowf) + "_"

    acc = _set_train(cnf)
    return acc


def _set_train(cnf):
    writer = MlflowWriter(cnf.experiment_name)
    writer.log_params_from_omegaconf_dict(cnf.model)

    dataloaders_dict, label_set = create_dataloader(cnf)
    cnf.model.label_num = len(label_set)

    model = Net(cnf)
    summary(model, (3, cnf.model.input_size, cnf.model.input_size))

    criterion = nn.CrossEntropyLoss()
    _learning_rate = float(cnf.train.learning_rate)
    optimizer = optim.RMSprop(model.parameters(), lr=_learning_rate)

    _sd_patience = cnf.train.sd_patience
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, mode='min', patience=_sd_patience)

    early_stopping = train_model(
        cnf, model, dataloaders_dict, criterion, optimizer, scheduler, writer)

    return 1 - early_stopping.acc()


def train_model(cnf, model, dataloaders_dict, criterion, optimizer, scheduler, writer):
    logger.info(f"Log output dir: {os.getcwd()}")
    # tensor_writer = tensorboard.SummaryWriter(
    #    log_dir=os.getcwd())  # hydraのoutputになっているはず。

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"{device}で学習を行います。")
    model.to(device)

    # ネットワークがある程度固定ならば高速化
    torch.backends.cudnn.benchmark = True

    # Early stopping
    early_stopping = EarlyStopping(cnf.train.early_stopping)
    _early_stop = False

    # モデルの保存に関して
    model_w_dir = os.path.join(os.getcwd(), cnf.model_dir)
    model_w_prefix = os.path.join(
        model_w_dir, cnf.model_weight_prefix)
    os.makedirs(model_w_dir, exist_ok=True)

    # scheduler
    learning_rate = float(optimizer.param_groups[0]['lr'])
    for epoch in range(cnf.train.num_epochs):
        logger.info(f"Epoch {epoch + 1} / {cnf.train.num_epochs}")

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            epoch_loss = 0.
            epoch_corrects = 0.

            # 未学習時の検証性能を確かめるため、epoch=0の訓練は省略
            if (epoch == 0) and (phase == 'train'):
                continue

            for inputs, labels in tqdm(dataloaders_dict[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # optimizer を初期化
                optimizer.zero_grad()

                # forwardを計算
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    epoch_loss += loss.item() * inputs.size(0)  # lossの合計を計算
                    epoch_corrects += torch.sum(preds == labels.data)

            # epochごとのlossと正解率を表示
            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects.double(
            ) / len(dataloaders_dict[phase].dataset)
            epoch_acc = float(epoch_acc)
            # lossを考慮して、learning rateを更新
            if phase == 'val':
                scheduler.step(epoch_loss)
                learning_rate = float(optimizer.param_groups[0]['lr'])

            # Early Stoping
            if phase == "val":
                _early_stop, best_model = early_stopping.check_loss(epoch_loss)

                if best_model:
                    model_weight_path = model_w_prefix + \
                        '_' + str(epoch) + '.pth'
                    early_stopping.best_model_param(
                        epoch=epoch, model_path=model_weight_path, acc=epoch_acc)
                    writer.log_step_metric(
                        f"best-loss", epoch_loss, step=epoch)
                    writer.log_step_metric(f"best-acc", epoch_acc, step=epoch)
                    torch.save(model.state_dict(), model_weight_path)

            logger.info(
                f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} lr: {learning_rate}")

            writer.log_step_metric(f"{phase}-loss", epoch_loss, step=epoch)
            writer.log_step_metric(f"{phase}-acc", epoch_acc, step=epoch)
        if _early_stop:
            model_weight_path = model_w_prefix + \
                '_' + str(epoch) + '_best.pth'
            torch.save(model.to('cpu').state_dict(), model_weight_path)
            logger.info(
                f"Early Stoping: {early_stopping.epoch()}, Loss: {early_stopping.loss()}, Acc: {early_stopping.acc()}")
            break

    return early_stopping


if __name__ == "__main__":
    train()
