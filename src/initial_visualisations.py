from data.data import CharacterIndexing, DataObject
import tensorflow as tf
import os
import pandas as pd
import matplotlib.pyplot as plt


models =["rnn", "lstm", "gru"]
sequence_lengths = [50,100,150,200]


callback_base_folder = "./src/models/callbacks"


def save_plot_for_model(model_name: str):
    for sequence in sequence_lengths:
        sequnce_folder = f"sequence_{sequence}"
        csv_tracker = os.path.join(callback_base_folder, model_name, sequnce_folder, "csv", "csv_tracker")

        csv_data = pd.read_csv(csv_tracker, names=["epoch", "loss", "val_loss"], header=0)
        print(csv_data)


        plt.plot(csv_data["epoch"], csv_data["loss"], label = f"sequence_{sequence}")


    plt.title(model_name)
    plt.legend()
    plt.savefig(f"./{model_name}")
    plt.clf()


for model in models : save_plot_for_model(model)












