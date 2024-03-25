import matplotlib.pyplot as plt
import os 
import pandas as pd



root_path = "./"
root_path = os.path.abspath(root_path)
base_model_path = os.path.join(root_path, "models")


def get_model_path(model_name: str):
    return os.path.join(base_model_path, model_name)



def read_columns_from_csv(model_name: str, column_names: list):
    model_path = get_model_path(model_name)
    csv_file = os.path.join(os.path.join(model_path, "csv_tracker", "csv_tracker"))
    d_frame= pd.read_csv(csv_file)[column_names] 

    return d_frame


def plot_errors(model_list: list, col_names: list, names_list: list):
    plt.figure(figsize=(10, 6))  # Adjust size of the plot

    for i, model in enumerate(model_list):
        err_vals = read_columns_from_csv(model, col_names)
        plt.plot(err_vals, label=names_list[i])  # Use the current index i for labels
    
    plt.title('Model Errors')  # Add title
    plt.xlabel('Iteration')  # Label x-axis
    plt.ylabel('Error Value')  # Label y-axis
    plt.legend()  # Show legend
    plt.grid(True)  # Add grid for better readability
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    plt.show()




models = ["W_P_T_S1.0", "W_P_T_S1.1", "W_P_T_S1.2", "W_P_T_S2.0", "W_P_T_M1.0"]#["W_P_T_S2.0", "W_P_T_M1.0"]
column_names = ["val_loss"]#, "val_loss"]
names = models#"Training"#["Training", "Validation"]

plot_errors(models,column_names,names)


