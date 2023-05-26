"""
The main script to run FPI Pruning
"""
from torch.utils.data import DataLoader
import os
import csv
import argparse
import copy
import time

import FPI.FPI_algo as FPI
from Preprocessing.Datasets import SegmentationDataSetNPZ
from Development.DevUtils import load_model_from_json_path
import configurations as c
import torch


#MODEL_JSON = "_results/saved_models/pruning-references-test_2021-03-29_08-29-00.json"pruning-UnetClassic-reference_2021-05-05_09-24-39.json
#MODEL_STATE_DICT = "_results/saved_models/pruning-references-test_2021-03-29_08-29-00.pt"

N_CLASSES = 1
# hyperparameters for pruning
parser = argparse.ArgumentParser()
parser.add_argument('--amount', type=float, default=0.2, help='pruning amount')
parser.add_argument('--method', type = str, default = "l1", help = 'pruning method' )
parser.add_argument('--arch', type= str, default ="TernausNet", help = "architecture" )
args = parser.parse_args()

def main():
    prune_amount = args.amount
    method = args.method
    arch = args.arch

    # choose either pretrained TernausNet or UNet
    if arch == "TernausNet":
        MODEL_JSON = "/data/project-gxb/johannes/innspector/_results/saved_models/pruning-references-test_2021-03-29_08-29-00.json"
        MODEL_STATE_DICT = "/data/project-gxb/johannes/innspector/_results/saved_models/pruning-references-test_2021-03-29_08-29-00.pt"
        model_name = "pruning-references-test_2021-03-29_08-29-00"
    else:
        MODEL_JSON = "/data/project-gxb/johannes/innspector/_results/saved_models/pruning-UnetClassic-reference_2021-05-05_09-24-39.json"
        MODEL_STATE_DICT = "/data/project-gxb/johannes/innspector/_results/saved_models/pruning-UnetClassic-reference_2021-05-05_09-24-39.pt"
        model_name = "pruning-UnetClassic-reference_2021-05-05_09-24-39"

    # prepare train and validation images and masks from INRIA segmentation dataset
    d_conf = c.get_data_config()["inria"]
    train_names = os.listdir(d_conf["train_dir"] + d_conf["image_folder"])
    val_names = os.listdir(d_conf["val_dir"] + d_conf["image_folder"])
    p_imgs_train = [d_conf["train_dir"] + d_conf["image_folder"] + n for n in train_names]
    p_msk_train = [d_conf["train_dir"] + d_conf["mask_folder"] + n for n in train_names]
    p_imgs_val = [d_conf["val_dir"] + d_conf["image_folder"] + n for n in val_names]
    p_msk_val = [d_conf["val_dir"] + d_conf["mask_folder"] + n for n in val_names]

    img_paths = p_imgs_val
    mask_paths = p_msk_val

    #initialize the dataset classes and dataloaders
    valset = SegmentationDataSetNPZ(img_paths=img_paths, mask_paths=mask_paths, p_flips=None, p_noise=None)
    valloader = DataLoader(valset, batch_size=64, shuffle=False, num_workers=8, pin_memory = True )
    trainset = SegmentationDataSetNPZ(img_paths=p_imgs_train, mask_paths=p_msk_train,
                                      p_flips=0.5, p_noise=0.5)
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)

    #load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_from_json_path(model_json_path=MODEL_JSON, load_state_dict=False)
    model.load_state_dict(torch.load(MODEL_STATE_DICT, map_location = device))
    model = model.to(device)

    # get layers and number of filters for pruning
    layers, n_filters = FPI.get_layers_for_pruning(model, N_CLASSES)

    metrics = ["jaccard score", "acc. seg."]


    start_time = time.time()
    print("-" * 150 +
          "\nPRUNING MODEL")

    # ürune either with default pruning or other pruning methods
    for layer in layers:
        if method != "l1-structured":
            FPI.FPI_prune(layer,layer.weight.shape[0], prune_amount,method,device)
        else:
            FPI.default_prune(layer,prune_amount)

    prune_time = time.time()-start_time
    print(f"  Pruning takes %s seconds" %(prune_time))

    # test validation performance of the pruned model
    run_metrics, performance_score = FPI.validate_model(model=model,
                                                        valloader=valloader,
                                                        criterion=torch.nn.BCELoss(reduction="mean"),
                                                        metrics=metrics,
                                                        device=device)
    flops_score = FPI.calculate_pruned_flops(model, relevant_layers=layers, shape=(1, 3, 256, 256), device=device)
    seg_acc = run_metrics["vl acc. seg."]
    meanIoU = run_metrics["vl jaccard score"]
    print(f"  > Validated performance = {round(performance_score, 4)}  / seg. acc. = {seg_acc} / meanIoU = {meanIoU} ")


    # further finetune pruned models
    max_epochs = 15
    pruned_model, fine_tuned_metrics, performance_score = FPI.train_model(model, trainloader, valloader, max_epochs, metrics)
    flops_score = FPI.calculate_pruned_flops(pruned_model, relevant_layers=layers, shape=(1, 3, 256, 256), device=device)
    n_params_layer, n_pruned_layer = FPI.calculate_pruned_params(layers)
    seg_acc = fine_tuned_metrics["vl acc. seg."][-1]
    meanIoU = fine_tuned_metrics["vl jaccard score"][-1]
    print(
        f"  > Fine tuned performance = {round(performance_score, 4)} / flops = {flops_score} / seg. acc. = {seg_acc} / meanIoU = {meanIoU} / ")

    # save pruned models
    save_path = copy.deepcopy(MODEL_STATE_DICT)
    save_path = save_path.replace(model_name,
                                  arch + "-pruning-amount-" + str(prune_amount) + "-layerwise-epochs-true" + str(max_epochs)
                                  + "-pruning-method-" + args.method )
    FPI.save_model(pruned_model,layers, save_path)

    # write the results into csv
    csv_columns = ['Pruning amount', 'Performance', 'FLOPS', 'Seg. Acc.', 'meanIoU','Pruned params', 'Pruned filters', 'Method']
    dict = [
        {'Pruning amount': prune_amount, 'Performance': round(performance_score, 4), 'FLOPS': flops_score,
         'Seg. Acc.': seg_acc, 'meanIoU': meanIoU, 'Pruned params': n_params_layer, 'Pruned filters': n_pruned_layer, 'Method' : method}
    ]

    csv_file = "/data/project-gxb/johannes/innspector/_results/FPI_prune_amount_params.csv"
    file_exists = os.path.isfile(csv_file)

    try:
        with open(csv_file, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, delimiter=',', fieldnames=csv_columns)
            if not file_exists:
                writer.writeheader()
            for data in dict:
                writer.writerow(data)
    except IOError:
        print("I/O error")


if __name__ == "__main__":
    main()