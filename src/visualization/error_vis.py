import matplotlib.pyplot as plt
import os 
import pandas as pd



root_path = "./"
root_path = os.path.abspath(root_path)
base_model_path = os.path.join(root_path, "models")


def read_error_vals(input_mod_name: str):
    model_path = os.path.join(base_model_path, input_mod_name)
    csv_file = os.path.join(os.path.join(model_path, "csv_tracker", "csv_tracker"))

    d_frame= pd.read_csv(csv_file)[["loss"]] 

    return d_frame


def plot_errors(model_list: list, names_list: list):
    for i, model in enumerate(model_list):
        err_vals = read_error_vals(model)
        plt.plot(err_vals, label = names_list[i])
    
    plt.legend()
    plt.show()



#plot_errors(["RNN100Seq256Emb512Dense", "LSTM100Seq256Emb512Dense"]
#            , ["RNN", "LSTM"])

plot_errors(["WebTransformer_CharInd_128e_256d_7h_5l", "WebTransformer_CharInd_128e_512d_8h_8l"]
            , ["1", "2"])
