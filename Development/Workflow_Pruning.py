import torch
import sklearn.metrics
import json
import time
import os
import sklearn.model_selection
import datetime
from torchsummary import summary

from configurations import get_train_config
from Models.ModelZoo import get_model_from_zoo
import Development.DevUtils as devU
from .LRSchedule import MyLRScheduler
import torch.nn.utils.prune as prune
class ThresholdPruning(prune.BasePruningMethod):
    PRUNING_TYPE = "unstructured"

    def __init__(self, threshold):
        self.threshold = threshold

    def compute_mask(self, tensor, default_mask):
        return torch.abs(tensor) > self.threshold


def compute_metrics(preds, targets, metrics):
    """
    Wrapper to flexible compute metrics directly on gpu
    See Development.DevUtils.py for implementations
    Note that some classification metrics may require to use torch.max() or something ...
    """
    metrics_funcs = {
        "acc.": devU.Accuracy(),
        "jaccard_sim": devU.JaccardSimilarity(reduction="mean"),
        "jaccard_score": devU.JaccardScore(threshold=0.5, reduction="mean"),
        "crossentropy": torch.nn.CrossEntropyLoss(weight=None, reduction="mean"),
        "bce": torch.nn.BCEWithLogitsLoss(weight=None, reduction="mean")
    }

    res_dict = {}
    for key in metrics:
        try:
            res_dict[key] = metrics_funcs[key](preds, targets)
        except:
            raise NotImplementedError(f"Error in compute_classification_metrics() for {key} " +
                                      "- make sure a metric function is implemented ... ")
    return res_dict


def get_optimizer(model_parameters, optim_info):
    if optim_info[0] == "adam":
        return torch.optim.Adam(model_parameters, **optim_info[1])
    elif optim_info[0] == "sgd":
        return torch.optim.SGD(model_parameters, **optim_info[1])
    else:
        raise ValueError("Optimizer not implemented in get_optimizer()")


def get_criterion(criterion_name):
    if criterion_name == "crossentropy":
        return torch.nn.CrossEntropyLoss(weight=None, reduction="mean")
    elif criterion_name == "mse":
        return torch.nn.MSELoss(reduction="mean")
    elif criterion_name == "jaccard_sim":
        return devU.JaccardSimilarity(reduction="mean")
    elif criterion_name == "bce-jaccard_sim":
        return devU.BCEJaccardSim(reduction="mean")
    elif criterion_name == "bce":
        return torch.nn.BCEWithLogitsLoss(weight=None, reduction="mean")
    else:
        raise ValueError("Criterion not implemented in get_criterion()")


# training function for classification problem ...
def train_model(model, model_path, trainloader, valloader, max_epochs,
                optimizer, criterion, metrics, lr_scheduler,
                early_stopping):

    # setup return variables
    res_dict = {
        "epoch": [],
        "tr loss": [],
        "vl loss": []
    }
    for m in metrics:
        res_dict["tr " + m] = []
        res_dict["vl " + m] = []

    model_dict = model.get_model_dict()
    model_dict["lr_scheduler"] = lr_scheduler
    min_val_loss = 100.0
    start_time = time.time()

    # only used if early_stopping == True
    n_early_stopping_epochs = get_train_config()["early_stopping_n_epochs"]
    early_stopping_ctr = 0

    # setup lr_schedulers if specified
    if lr_scheduler == "anneal":
        anneal_lr_min = get_train_config()["anneal_lr_min"]
        anneal_lr_T0 = get_train_config()["anneal_lr_T0"]
        anneal_lr_Tmult = get_train_config()["anneal_lr_Tmult"]
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                         T_0=anneal_lr_T0,
                                                                         T_mult=anneal_lr_Tmult,
                                                                         eta_min=anneal_lr_min, verbose=False)
        # use early stopping variables to check if a cycle is completed
        n_early_stopping_epochs = anneal_lr_T0
        # make sure a min. of one cycle is run
        if max_epochs < anneal_lr_T0:
            max_epochs = anneal_lr_T0
            print("> increased max_epochs to", anneal_lr_T0, "since lr_scheduler == 'anneal'")
    elif lr_scheduler == "stepwise":
        patience = get_train_config()["stepwise_lr_patience"]
        stepwise_lr_min = get_train_config()["stepwise_lr_min"]
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               mode="min",
                                                               factor=0.1,
                                                               patience=patience,
                                                               min_lr=stepwise_lr_min, verbose=False)
    elif lr_scheduler == "custom":
        step_sizes = [51, 101, 131, 161, 191, 221, 251, 281]
        customLR = MyLRScheduler( 0.1, 5, step_sizes)

    print("-"*150 +
          "\nTRAINING STARTED")
    # ============================================================================================================
    # get available devices for parallelization
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("We are using", torch.cuda.device_count(), "GPUs!", " Device = ", device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)
    criterion.to(device)


    for epoch in range(max_epochs):
        # setup some more variables to save train stats
        run_loss_tr = 0.0
        run_loss_val = 0.0
        run_metrics = {}
        for m in metrics:
            run_metrics["tr " + m] = 0.0
            run_metrics["vl " + m] = 0.0

        eptime = time.time()

        lr_log = customLR.get_lr(epoch)
        # set the optimizer with the learning rate
        # This can be done inside the MyLRScheduler
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_log
        # ========================================================================================================
        # train on trainloader

        #if epoch > 5 and epoch % 3 == 0:
        if epoch >= 0:
            module1 = model.level5._modules['4'].conv
            print(
                "Sparsity in conv1.weight: {:.2f}%".format(
                    100. * float(torch.sum(module1.weight == 0))
                    / float(module1.weight.nelement())
                )
            )

            parameters_to_prune = [(module1, "weight")]
            prune.global_unstructured(
                parameters_to_prune, pruning_method=ThresholdPruning, threshold=0.1
            )
            print(
                "Sparsity in conv1.weight after pruning: {:.2f}%".format(
                    100. * float(torch.sum(module1.weight == 0))
                    / float(module1.weight.nelement())
                )
            )


        model.train()
        for i, (data, labels) in enumerate(trainloader):
            data, labels = data.to(device), labels.to(device)

            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = model(data).to(device)
                labels = labels.type(torch.LongTensor).to(device)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            metrics_tr = compute_metrics(preds=outputs, targets=labels, metrics=metrics)
            run_loss_tr += loss.item()
            for m in metrics:
                run_metrics["tr " + m] += metrics_tr[m].item()

        # ========================================================================================================
        # evaluate on valloader
        model.eval()
        for i, (data, labels) in enumerate(valloader):
            data, labels = data.to(device), labels.to(device)

            optimizer.zero_grad()
            with torch.set_grad_enabled(False):
                outputs = model(data).to(device)
                labels = labels.type(torch.LongTensor).to(device)
                loss = criterion(outputs, labels)

            metrics_val = compute_metrics(preds=outputs, targets=labels, metrics=metrics)
            run_loss_val += loss.item()
            for m in metrics:
                run_metrics["vl " + m] += metrics_val[m].item()

        # ========================================================================================================
        # step lr schedulers if specified
        if lr_scheduler == "anneal":
            scheduler.step()
            early_stopping_ctr += 1
        elif lr_scheduler == "stepwise":
            scheduler.step(run_loss_val)


        # ========================================================================================================
        # compute metrics for epoch, append to res_dict and print epoch stats
        res_dict["epoch"].append(epoch+1)
        res_dict["tr loss"].append(run_loss_tr/len(trainloader))
        res_dict["vl loss"].append(run_loss_val/len(valloader))
        for m in metrics:
            res_dict["tr " + m].append(run_metrics["tr " + m]/len(trainloader))
            res_dict["vl " + m].append(run_metrics["vl " + m]/len(valloader))

        for key in res_dict.keys():
            value = str(round(res_dict[key][-1], 3))
            pad = 5 - len(value)
            value += pad*" "
            print("{}: {} | ".format(key, value), end="")
        print(f" {round(time.time()-eptime)} seconds")

        # ========================================================================================================
        # saving best model and abort if early stopping is met. if cosine annealing with warm
        # restarts is used, this is only done after a cycle is finished.
        if lr_scheduler != "anneal":
            # save best performing model and perform early stopping if specified
            if res_dict["vl loss"][-1] < min_val_loss:
                min_val_loss = res_dict["vl loss"][-1]

                torch.save(model.state_dict(), model_path)
                model_dict["state_dict_path"] = model_path
                model_dict["best_epoch"] = epoch+1
                with open(model_path.replace(".pt", ".json"), "w") as mf:
                    json.dump(model_dict, mf)
                early_stopping_ctr = 0
            else:
                early_stopping_ctr += 1

            # abort if early stopping criteria met
            if early_stopping and (early_stopping_ctr >= n_early_stopping_epochs):
                break
        elif (lr_scheduler == "anneal") and (early_stopping_ctr == n_early_stopping_epochs):
            # only save model after an annealing cycle is finished to avoid noise in performance
            # and allow to use the epoch for re-training of the model
            # if early stopping is used, training is stopped after a cycle without improvement
            if res_dict["vl loss"][-1] < min_val_loss:
                min_val_loss = res_dict["vl loss"][-1]
                torch.save(model.state_dict(), model_path)
                model_dict["state_dict_path"] = model_path
                model_dict["best_epoch"] = epoch + 1
                early_stopping_ctr = 0
                with open(model_path.replace(".pt", ".json"), "w") as mf:
                    json.dump(model_dict, mf)
            else:
                if early_stopping:
                    break
        else:
            pass

    print(f"TRAINING FINISHED after {round((time.time()-start_time)/60, 2)} minutes")
    print("-"*150)
    return model_dict, res_dict


def run_hyperparameters(hps_name, grid_dict, trainset, valset):
    # before making parameter grid, catch undesired behaviour of sklearn if one parameter is not in a list
    for v in grid_dict.values():
        if type(v) != list:
            raise ValueError("All values of grid_dict should be lists," +
                             "to avoid unintended behaviour of sklearn ParameterGrid()")
    run_params = list(sklearn.model_selection.ParameterGrid(grid_dict))
    print("="*150)
    print("RUNNING {} HYPERPARAMETER COMBINATIONS".format(len(run_params)))

    # check if dataloaders can be declared once or have to be declared in every run
    if len(grid_dict["batchsize"]) == 1:
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=grid_dict["batchsize"][0],
                                                  shuffle=True, num_workers=2, drop_last=False)
        valloader = torch.utils.data.DataLoader(dataset=valset, batch_size=1, shuffle=False,
                                                num_workers=2, drop_last=False)

    # loop over hyperparameter combinations
    for params in run_params:

        # set up administrative variables
        idx = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
        r_name = "{}_{}".format(hps_name, str(idx))
        model_path = get_train_config()["export_path"] + r_name + ".pt"

        overview_dict = {
            "run id": r_name,
            "mode": params["mode"],
            "model name": params["model"][0],
            "model args": str(params["model"][1]),
            "model path": model_path,
            "optimizer": params["optimizer"][0],
            "optimizer args": str(params["optimizer"][1]),
            "criterion": params["criterion"],
            "batchsize": params["batchsize"],
            "lr scheduler": params["lr_scheduler"],
            "max. epochs": params["max_epochs"],
            "early stopping": params["early_stopping"]
        }

        print("="*150)
        print("CURRENT HYPERPARAMETRS")
        for k in overview_dict.keys():
            print("\t{}: {}".format(k, overview_dict[k]))

        # in case dataloaders were not declared above, build them now
        if len(grid_dict["batchsize"]) > 1:
            trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=params["batchsize"], shuffle=True,
                                                      num_workers=0, drop_last=False)
            valloader = torch.utils.data.DataLoader(dataset=valset, batch_size=1, shuffle=False,
                                                    num_workers=0, drop_last=False)

        # get fundamentals
        model = get_model_from_zoo(params["model"])
        summary(model,(3,256,256))
        optimizer = get_optimizer(model.parameters(), params["optimizer"])
        criterion = get_criterion(params["criterion"])

        # perform a train run
        model_d, res_d = train_model(model=model, model_path=model_path,
                                     trainloader=trainloader, valloader=valloader,
                                     max_epochs=params["max_epochs"],
                                     optimizer=optimizer,
                                     criterion=criterion,
                                     metrics=get_train_config()[params["mode"]]["train_metrics"],
                                     lr_scheduler=params["lr_scheduler"],
                                     early_stopping=params["early_stopping"])

        # avoid risk of memory overflow
        del model, optimizer, criterion

        # export results
        overview_dict["best epoch"] = model_d["best_epoch"]
        with open(model_path.replace(".pt", "_hp-overview.json"), "w") as mf:
            json.dump(overview_dict, mf)

        with open(model_path.replace(".pt", "_train-logs.json"), "w") as mf:
            json.dump(res_d, mf)

        print("EXPORTED RESULTS")


if __name__ == "__main__":
    pass
