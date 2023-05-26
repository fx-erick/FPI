import torch
import sklearn.metrics
import json
import time
import os
import sklearn.model_selection
import datetime
import torch.nn.utils.prune as prune
import copy
from torchsummary import summary


from configurations_segmentation import get_train_config
#from configurations import get_train_config
from Models.ModelZoo import get_model_from_zoo
import Development.DevUtils as devU
from collections import OrderedDict
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
        "acc": devU.Accuracy(),
        "jaccard_sim.": devU.JaccardSimilarity(reduction="mean"),
        "jaccard_score": devU.JaccardScore(threshold=0.5, reduction="mean"),
        "crossentropy": torch.nn.CrossEntropyLoss(weight=None, reduction="mean"),
        "bce": torch.nn.BCELoss(weight=None, reduction="mean"),
        "acc. seg.": devU.AccuracySegmentation(threshold=0.5),
        "ruzicka": devU.RuzickaSimilarity(reduction="mean")
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
    elif criterion_name == "jaccard loss":
        return devU.JaccardSimilarity(reduction="mean")
    elif criterion_name == "bce-jaccard loss":
        return devU.BCEJaccardSim(reduction="mean")
    elif criterion_name == "bce":
        return torch.nn.BCELoss(weight=None, reduction="mean")
    elif criterion_name == "ruzicka":
        return devU.RuzickaSimilarity(reduction="mean")
    elif criterion_name == "bce-ruzicka":
        return devU.BCERuzickaSim(reduction="mean")
    else:
        raise ValueError("Criterion not implemented in get_criterion()")


# training function with various options ...
def train_model(model, model_path, trainloader, valloader, max_epochs,
                optimizer_def, criterion, metrics, lr_scheduler,
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

    # get available devices for parallelization
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("We are using", torch.cuda.device_count(), "GPUs!")
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)
    criterion.to(device)    

    # build optimizer after model is on gpu (some need it to be ... )
    optimizer = get_optimizer(model.parameters(), optimizer_def)

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

    print("-"*150 +
          "\nTRAINING STARTED")
    # ============================================================================================================

    for epoch in range(max_epochs):
        # setup some more variables to save train stats
        run_loss_tr = 0.0
        run_loss_val = 0.0
        run_metrics = {}
        for m in metrics:
            run_metrics["tr " + m] = 0.0
            run_metrics["vl " + m] = 0.0

        eptime = time.time()
        # ========================================================================================================
        # train on trainloader
        model.train()
        for data, labels in trainloader:
            data, labels = data.to(device), labels.to(device)

            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = model(data).to(device)

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
        for data, labels in valloader:
            data, labels = data.to(device), labels.to(device)

            optimizer.zero_grad()
            with torch.set_grad_enabled(False):
                outputs = model(data).to(device)

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
                    mf.write(json.dumps(model_dict, indent=4))
                early_stopping_ctr = 0
            else:
                early_stopping_ctr += 1

            torch.save(model.state_dict(), model_path.replace(".pt", "_last_epoch.pt"))  # to also obtain the parameters from the last epoch

            # abort if early stopping criteria met
            if early_stopping and (early_stopping_ctr >= n_early_stopping_epochs):
                break
        elif (lr_scheduler == "anneal") and (early_stopping_ctr == n_early_stopping_epochs):
            # only save model after an annealing cycle is finished to avoid noise in performance
            # and allow to use the epoch for re-training of the model
            # if early stopping is used, training is stopped after a cycle without improvement

            torch.save(model.state_dict(), model_path.replace(".pt", "_last_epoch.pt"))  # to also obtain the parameters from the last epoch

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
    del optimizer  # paranoia of memory leakage
    return model_dict, res_dict


def run_single_hyperparameter_combination_from_scratch(hps_name, params, trainloader, valloader):
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

    print("=" * 150)
    print("CURRENT HYPERPARAMETERS - TRAINING FROM SCRATCH")
    for k in overview_dict.keys():
        print("\t{}: {}".format(k, overview_dict[k]))

    # get fundamentals
    model = get_model_from_zoo(params["model"])
    criterion = get_criterion(params["criterion"])

    # perform a train run
    model_d, res_d = train_model(model=model, model_path=model_path,
                                 trainloader=trainloader, valloader=valloader,
                                 max_epochs=params["max_epochs"],
                                 optimizer_def=params["optimizer"],
                                 criterion=criterion,
                                 metrics=get_train_config()[params["mode"]]["train_metrics"],
                                 lr_scheduler=params["lr_scheduler"],
                                 early_stopping=params["early_stopping"])

    # avoid risk of memory overflow
    del model, criterion

    # export results
    overview_dict["best epoch"] = model_d["best_epoch"]
    with open(model_path.replace(".pt", "_hp-overview.json"), "w") as mf:
        mf.write(json.dumps(overview_dict, indent=4))

    with open(model_path.replace(".pt", "_train-logs.json"), "w") as mf:
        mf.write(json.dumps(res_d, indent=4))

    print("EXPORTED RESULTS")
    return model_d, res_d


def run_existing_model(model, hps_name, params, trainloader, valloader, save_init_model=False):
    # set up administrative variables
    idx = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
    r_name = "{}_{}".format(hps_name, str(idx))
    model_path = get_train_config()["export_path"] + r_name + ".pt"

    model_info = model.get_model_dict()
    model_args = {}
    for key in model_info.keys():
        if key != "type":
            model_args[key] = model_info[key]

    if save_init_model:
        torch.save(model.state_dict(), model_path.replace(".pt", "_init.pt"))

    overview_dict = {
        "run id": r_name,
        "mode": params["mode"],
        "model name": model_info["type"],
        "model args": model_args,
        "model path": model_path,
        "optimizer": params["optimizer"][0],
        "optimizer args": str(params["optimizer"][1]),
        "criterion": params["criterion"],
        "batchsize": params["batchsize"],
        "lr scheduler": params["lr_scheduler"],
        "max. epochs": params["max_epochs"],
        "early stopping": params["early_stopping"]
    }

    if list(params.keys()).__contains__("pruning method"):
        overview_dict["pruning method"] = params["pruning method"]
        overview_dict["pruning amount"] = params["pruning amount"]
        overview_dict["pruning iter"] = params["pruning iter"]

    print("=" * 150)
    print("CURRENT HYPERPARAMETERS - TRAINING EXISTING MODEL")
    for k in overview_dict.keys():
        print("\t{}: {}".format(k, overview_dict[k]))

    # get fundamentals
    criterion = get_criterion(params["criterion"])

    # perform a train run
    model_d, res_d = train_model(model=model, model_path=model_path,
                                 trainloader=trainloader, valloader=valloader,
                                 max_epochs=params["max_epochs"],
                                 optimizer_def=params["optimizer"],
                                 criterion=criterion,
                                 metrics=get_train_config()[params["mode"]]["train_metrics"],
                                 lr_scheduler=params["lr_scheduler"],
                                 early_stopping=params["early_stopping"])

    # avoid risk of memory overflow
    del model, criterion

    # export results
    overview_dict["best epoch"] = model_d["best_epoch"]
    with open(model_path.replace(".pt", "_hp-overview.json"), "w") as mf:
        json.dump(overview_dict, mf)

    with open(model_path.replace(".pt", "_train-logs.json"), "w") as mf:
        json.dump(res_d, mf)

    print("EXPORTED RESULTS")
    return model_d, res_d


def build_dataloaders_and_param_grid(grid_dict, trainset, valset):
    # before making parameter grid, catch undesired behaviour of sklearn if one parameter is not in a list
    for v in grid_dict.values():
        if type(v) != list:
            raise ValueError("All values of grid_dict should be lists," +
                             "to avoid unintended behaviour of sklearn ParameterGrid()")
    run_params = list(sklearn.model_selection.ParameterGrid(grid_dict))
    print("=" * 150)
    print("RUNNING {} HYPERPARAMETER COMBINATIONS".format(len(run_params)))

    # check if dataloaders can be declared once or have to be declared in every run
    if len(grid_dict["batchsize"]) != 1:
        raise ValueError("Sorry, only a single batchsize per hyperparameter search is allowed currently")

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=grid_dict["batchsize"][0],
                                              shuffle=True, num_workers=1, drop_last=False)
    valloader = torch.utils.data.DataLoader(dataset=valset, batch_size=4, shuffle=False,
                                            num_workers=1, drop_last=False)

    return run_params, trainloader, valloader


def run_hyperparameters(hps_name, grid_dict, trainset, valset):
    # get parameter grid and datalaoders
    run_params, trainloader, valloader = build_dataloaders_and_param_grid(grid_dict=grid_dict,
                                                                          trainset=trainset,
                                                                          valset=valset)

    # loop over hyperparameter combinations
    for params in run_params:
        _, _ = run_single_hyperparameter_combination_from_scratch(hps_name=hps_name, params=params,
                                                                  trainloader=trainloader, valloader=valloader)


def run_hyperparameters_with_pruning_steps(hps_name,
                                           grid_dict,
                                           trainset,
                                           valset,
                                           pruning_method,
                                           n_pruning_iters,
                                           amount,
                                           shrutika_prune_args=None):
    # get parameter grid and datalaoders
    run_params, trainloader, valloader = build_dataloaders_and_param_grid(grid_dict=grid_dict,
                                                                          trainset=trainset,
                                                                          valset=valset)

    if len(run_params) > 1:
        raise ValueError("more than one hyperparameter combination is currently not supported for pruning")
    params = run_params[0]

    if grid_dict["model"][0][0] != "UNetVGGbase" and pruning_method != "theresa":
        raise ValueError("run_hyperparameters_with_pruning_steps is only intended for UNetVGGbase models ..." +
                         "you should probably check/adapt its implementation in Workflow.py")

    if shrutika_prune_args is not None:
        shrutika_prune_args["prune"] = amount

    # run model with newly initialized weights
    model_d, _ = run_single_hyperparameter_combination_from_scratch(hps_name=hps_name + "_pr-Base", params=params,
                                                                    trainloader=trainloader, valloader=valloader)

    for i in range(0, n_pruning_iters):
        model = devU.load_model_from_json_dict(model_d)

        conv_layers = []
        for m in model.down_blocks.modules():
            if isinstance(m, torch.nn.Conv2d):
                conv_layers.append(m)
        for m in model.latent.modules():
            if isinstance(m, torch.nn.Conv2d):
                conv_layers.append(m)
        for m in model.up_blocks.modules():
            if isinstance(m, torch.nn.Conv2d):
                conv_layers.append(m)

        if pruning_method == "theresa":
            model.set_own_weights_below_threshold_to_zero(threshold=amount)

        elif pruning_method == "l1_unstructured":
            print(f"\n>pruning with pytorch l1 unstructured, {amount*100} % of parameters are pruned\n")
            for layer in conv_layers:
                prune.l1_unstructured(layer, name="weight", amount=amount)
                #prune.remove(layer, "weight")

        elif pruning_method == "random_unstructured":
            print(f"\n>pruning with pytorch random unstructured, {amount*100} % of parameters are pruned\n")
            for layer in conv_layers:
                prune.random_unstructured(layer, name="weight", amount=amount)
                #prune.remove(layer, "weight")

        elif pruning_method == "l1_structured":
            print(f"\n>pruning with pytorch l1 structured, {amount*100} % of parameters are pruned\n")
            for layer in conv_layers:
                prune.ln_structured(layer, name="weight", amount=amount, n=1, dim=0)
                #prune.remove(layer, "weight")

        elif pruning_method == "random_structured":
            print(f"\n>pruning with pytorch random structured, {amount*100} % of parameters are pruned\n")
            for layer in conv_layers:
                prune.random_structured(layer, name="weight", amount=amount, dim=0)
                #prune.remove(layer, "weight")

        elif pruning_method == "shrutika":
            print(f"\n>pruning with shrutika pruning, {amount * 100} % of filters are pruned\n")
            for layer in conv_layers:
                devU.shrutika_prune(layer, shrutika_prune_args)

        params["pruning method"] = pruning_method
        params["pruning amount"] = amount
        params["pruning iter"] = i+1

        model_d, _ = run_existing_model(model=model, hps_name=hps_name + f"_pr-{i+1}", params=params,
                                        trainloader=trainloader, valloader=valloader)


def prune_conv_layers_of_model(model, pruning_method, block, type, amount, shrutika_prune_args=None):
    conv_layers = []
    parameters_to_prune = []

    n_weights = 0
    sum_zeros = 0
    
    if type == "down":
    	module1 = model.down_blocks
    elif type == "up":
        module1 = model.up_blocks
    elif type == "latent":
        module1 = model.latent
     
    module1 = module1[block].modules()

    for m in module1:
        if isinstance(m, torch.nn.Conv2d):
            parameters_to_prune.append((m,"weight"))
            n_weights += float(m.weight.nelement())
            sum_zeros += float(torch.sum(m.weight == 0))

    print("Sparsity in conv weights: {:.2f}%".format(100. * sum_zeros / n_weights))
    '''
    for m in model.latent.modules():
        if isinstance(m, torch.nn.Conv2d):
            conv_layers.append(m)
    for m in model.up_blocks.modules():
        if isinstance(m, torch.nn.Conv2d):
            conv_layers.append(m)

    '''

    if pruning_method == "threshold":
        print(f"\n>pruning with threshold pruning with threshold = 0.02\n")
        prune.global_unstructured(
                parameters_to_prune, pruning_method=ThresholdPruning, threshold=0.02
            )

    
    elif pruning_method == "l1_unstructured":
        print(f"\n>pruning with pytorch l1 unstructured, {amount * 100} % of parameters are pruned\n")
        prune.global_unstructured(
                parameters_to_prune, pruning_method=prune.L1Unstructured, amount=amount
            )

     #prune.l1_unstructured(parameters_to_prune, amount=amount)

    elif pruning_method == "random_unstructured":
        print(f"\n>pruning with pytorch random unstructured, {amount * 100} % of parameters are pruned\n")
        for layer in conv_layers:
            prune.random_unstructured(layer, name="weight", amount=amount)

    elif pruning_method == "l1_structured":
        print(f"\n>pruning with pytorch l1 structured, {amount * 100} % of parameters are pruned\n")
        for layer in conv_layers:
            prune.ln_structured(layer, name="weight", amount=amount, n=1, dim=0)

    elif pruning_method == "random_structured":
        print(f"\n>pruning with pytorch random structured, {amount * 100} % of parameters are pruned\n")
        for layer in conv_layers:
            prune.random_structured(layer, name="weight", amount=amount, dim=0)

    elif pruning_method == "shrutika":
        print(f"\n>pruning with shrutika pruning, {amount * 100} % of filters are pruned\n")
        for layer in conv_layers:
            devU.shrutika_prune(layer, amount, shrutika_prune_args)


    sum_zeros = 0
    if type == "down":
    	module1 = model.down_blocks
    elif type == "up":
        module1 = model.up_blocks
    elif type == "latent":
        module1 = model.latent
     
    module1 = module1[block].modules()

    for m in module1:
        if isinstance(m, torch.nn.Conv2d):
            sum_zeros += float(torch.sum(m.weight == 0))

    print("Sparsity in conv weights: {:.2f}%".format(100. * sum_zeros / n_weights))


def validate_model(model, params, valloader):
    # get available devices for parallelization
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("We are using", torch.cuda.device_count(), "GPUs!")
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)

    # get and declare stuff
    criterion = get_criterion(params["criterion"])
    metrics = get_train_config()[params["mode"]]["inference_metrics"]
    run_loss_val = 0.0
    run_metrics = {
        "vl loss": 0.0
    }
    for m in metrics:
        run_metrics["vl " + m] = 0.0

    res_dict = {}

    # ========================================================================================================
    # evaluate on valloader
    model.eval()
    for data, labels in valloader:
        data, labels = data.to(device), labels.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(data)
            loss = criterion(outputs, labels)

        metrics_val = compute_metrics(preds=outputs, targets=labels, metrics=metrics)
        run_loss_val += loss.item()
        for m in metrics:
            run_metrics["vl " + m] += metrics_val[m].item()

    # ========================================================================================================
    # compute metrics, append to res_dict and print stats
    res_dict["vl loss"] = run_loss_val / len(valloader)
    for m in metrics:
        res_dict["vl " + m] = run_metrics["vl " + m] / len(valloader)

    for key in res_dict.keys():
        value = str(round(res_dict[key], 3))
        pad = 5 - len(value)
        value += pad * " "
        print("{}: {} | ".format(key, value), end="")
    print()
    return res_dict


def run_single_pruning_experiment(hps_name, retrain_grid_dict, model_json_path, model_add_weights, pruning_spec, trainset, valset):

    res_dict = {}

    run_params, trainloader, valloader = build_dataloaders_and_param_grid(grid_dict=retrain_grid_dict,
                                                                          trainset=trainset,
                                                                          valset=valset)

    if len(run_params) > 1:
        raise ValueError("Sorry currently only a single hp-combination is supported, please adapt run_single_pruning_experiment()")

    # get model and perform initial validation
    print("-" * 150 + "\nINITIAL VALIDATION")
    #model = devU.load_model_from_json_path(model_json_path=model_json_path)
    model = get_model_from_zoo(model_add_weights)
    state_dict_path = model_json_path.replace(".json", ".pt")
    state_dict = torch.load(state_dict_path, map_location=torch.device('cpu'))

    #TODO!!!
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if "up_block.1" not in k and "final" not in k :
            if k[-6:] == "weight":
                name = k[:-7] + ".conv.weight"
            elif k[-4:] == "bias":
                name = k[:-5] + ".conv.bias"
        else:
            name = k
        new_state_dict[name] = v

    for name, param in model.named_parameters():
        if "weight_vector" in name:
            new_state_dict[name] = param.data

    model.load_state_dict(new_state_dict)

    summary(model,(3,256,256))
    model.eval()
    res_dict["base_vl"] = validate_model(model=model, params=run_params[0], valloader=valloader)
    print("-"*150)

    # prune model and perform initial validation
    
    prune_conv_layers_of_model(model=model,
                               pruning_method=pruning_spec["method"],
                               block = pruning_spec["block"],
                               type = pruning_spec["type"],
                               amount=pruning_spec["amount"],
                               shrutika_prune_args=pruning_spec["shrutika_prune_args"])

    

    print("-" * 150 + "\nPRUNED VALIDATION")
    res_dict["pruned_vl"] = validate_model(model=model, params=run_params[0], valloader=valloader)
    print("-" * 150)



    # retrain model and validate again
    model.train()
    model_d, run_logs = run_existing_model(model=model, hps_name=hps_name + "_retrained", params=run_params[0],
                                           trainloader=trainloader, valloader=valloader, save_init_model=True)

    res_dict["pruned_retrained_vl_min_loss_epoch"] = {}
    res_dict["pruned_retrained_vl_last_epoch"] = {}
    for k, v in run_logs.items():
        res_dict["pruned_retrained_vl_min_loss_epoch"][k] = v[model_d["best_epoch"]-1]
        res_dict["pruned_retrained_vl_last_epoch"][k] = v[-1]

    res_dict_path = model_d["state_dict_path"].replace("_retrained_", "_results_").replace(".pt", ".json")
    with open(res_dict_path, "w") as file:
        file.write(json.dumps(res_dict, indent=4))


if __name__ == "__main__":
    pass
