import pandas as pd
import os
import datetime
import matplotlib.pyplot as plt


def export_to_doc_file(doc_file_path, overview_dict, res_dict):
    """
    export information gathered during model selection to MS Excel file
    :param doc_file_path: (str) path to .xlsx file
    :param overview_dict: (dict of scalars) information to append to overview sheet
    :param res_dict: (dict of 1D lists) training history
    """

    # generate timestamp to use as identifier
    timestamp = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
    overview_dict["date"] = timestamp

    if os.path.exists(doc_file_path) is False:
        with pd.ExcelWriter(doc_file_path) as writer:
            pd.DataFrame(columns=overview_dict.keys()).to_excel(writer, sheet_name="Overview", index=False)

    # read excel to dict of dataframes and append new information
    doc_file = pd.read_excel(doc_file_path, sheet_name=None)
    doc_file["Overview"] = doc_file["Overview"].append(overview_dict, ignore_index=True)

    # check for duplicate keys/run_ids before inserting res_dict
    if doc_file.__contains__(overview_dict["run id"]):
        print("ATTENTION: run id '", overview_dict["run id"],  "' is already in use!",
              "Timestamp '", timestamp, "' is used as run id instead.")
        doc_file[timestamp] = pd.DataFrame(res_dict)
    else:
        doc_file[overview_dict["run id"]] = pd.DataFrame(res_dict)

    # renaming existing doc_file as a backup
    os.rename(doc_file_path, doc_file_path.split(".xlsx")[0] + "__" + timestamp + ".xlsx")

    # save modified dict of dataframes to excel
    with pd.ExcelWriter(doc_file_path) as writer:
        for key in doc_file:
            doc_file[key].to_excel(writer, sheet_name=key, index=False)


def plot_train_hist(path_name, res_dict, plot_name, epoch_key="epoch"):
    """
    creates and saves 2 subplots plots of training curves (loss/metric)
    :param path_name: (str) path to save plots to
    :param res_dict: (dict of 1D arrays) trainings history
    :param plot_name: (str) name for the plot
    :param epoch_key: (str) exact key to find values for x-axis in res_dict
    """

    # create figure
    plt.figure(figsize=(10, 7))

    # create subplot 1 with loss histories
    plt.subplot(211)
    plt.title("training curves - '{}'".format(plot_name))
    i = 0
    legend = []
    for key in res_dict:
        if key.__contains__(" loss"):
            if key.__contains__("tr loss"):
                linestyle = "-"
            else:
                linestyle = "--"
            plt.plot(res_dict[epoch_key], res_dict[key], label=key, linestyle=linestyle)
            legend.append(key)
            i += 1
    plt.ylabel("loss")
    plt.legend(legend, loc="upper right")

    # create subplot 2 with metric histories
    plt.subplot(212)
    i = 0
    legend = []
    for key in res_dict:
        if (key.__contains__(" loss") is False) and (key != "epoch"):
            if key.__contains__("tr "):
                linestyle = "-"
            else:
                linestyle = "--"
            plt.plot(res_dict[epoch_key], res_dict[key], label=key, linestyle=linestyle)
            legend.append(key)
            i += 1
    plt.xlabel("epoch")
    plt.ylabel("metric")
    plt.legend(legend, loc="lower right")

    # save and show plot
    timestamp = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
    plt.savefig(fname="{}/{}_{}.png".format(path_name, plot_name, timestamp))
    # plt.show()


if __name__ == "__main__":
    # info()

    # for debugging purposes:
    import random
    a = {
        "run_id": "run_13",
        "model_name": "Model_Y",
        "optimizer": "Adam-lr:0.01-beta:0.99",
        "loss_function": "MSE"
    }
    b = {
        "epoch": range(100),
        "tr loss": [random.random() for x in range(100)],
        "vl loss": [random.random() for x in range(100)],
        "tr acc.": [random.random() for x in range(100)],
        "vl acc.": [random.random() for x in range(100)],
        "tr f1-score": [random.random() for x in range(100)],
        "vl f1-score": [random.random() for x in range(100)]
    }

    # export_to_doc_file("../_results/doc_files/doc_file__2020-11-06_15-18-48.xlsx", a, b)
    # plot_train_hist("../_results/training_histories", b, a["run_id"])
