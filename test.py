import pandas as pd
import matplotlib.pyplot as plt

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def generate_comparison_plots(pytorch_csv, tensorflow_csv, jax_csv, output_plot_path):
    """
    Reads CSV files for PyTorch, TensorFlow, and JAX training metrics and generates
    comparison plots with bar graph for time taken and line graphs for other metrics.

    Args:
        pytorch_csv (str): Path to the PyTorch metrics CSV file.
        tensorflow_csv (str): Path to the TensorFlow metrics CSV file.
        jax_csv (str): Path to the JAX metrics CSV file.
        output_plot_path (str): Path to save the generated plot image.
    """
    try:
        # Load CSV files
        pytorch_data = pd.read_csv(pytorch_csv)
        tensorflow_data = pd.read_csv(tensorflow_csv)
        jax_data = pd.read_csv(jax_csv)

        # Create a tiled plot
        fig, axs = plt.subplots(2, 2, figsize=(14, 12))

        # Time per epoch as bar graph
        bar_width = 0.2  # Width of each bar
        epochs = np.arange(len(pytorch_data['Epoch']))  # Epoch indices

        axs[0, 0].bar(epochs - bar_width, pytorch_data['Time (s)'], bar_width, label='PyTorch')
        axs[0, 0].bar(epochs, tensorflow_data['Time (s)'], bar_width, label='TensorFlow')
        axs[0, 0].bar(epochs + bar_width, jax_data['Time (s)'], bar_width, label='JAX')

        axs[0, 0].set_title('Time Taken Per Epoch')
        axs[0, 0].set_xlabel('Epoch')
        axs[0, 0].set_ylabel('Time (s)')
        axs[0, 0].set_xticks(epochs)
        axs[0, 0].set_xticklabels(pytorch_data['Epoch'])
        axs[0, 0].grid(True)
        axs[0, 0].legend()

        # Training loss per epoch
        axs[0, 1].plot(pytorch_data['Epoch'], pytorch_data['Train Loss'], marker='o', label='PyTorch')
        axs[0, 1].plot(tensorflow_data['Epoch'], tensorflow_data['Train Loss'], marker='o', label='TensorFlow')
        axs[0, 1].plot(jax_data['Epoch'], jax_data['Train Loss'], marker='o', label='JAX')
        axs[0, 1].set_title('Training Loss Per Epoch')
        axs[0, 1].set_xlabel('Epoch')
        axs[0, 1].set_ylabel('Loss')
        axs[0, 1].grid(True)
        axs[0, 1].legend()

        # Training accuracy per epoch
        axs[1, 0].plot(pytorch_data['Epoch'], pytorch_data['Train Accuracy'], marker='o', label='PyTorch')
        axs[1, 0].plot(tensorflow_data['Epoch'], tensorflow_data['Train Accuracy'], marker='o', label='TensorFlow')
        axs[1, 0].plot(jax_data['Epoch'], jax_data['Train Accuracy'], marker='o', label='JAX')
        axs[1, 0].set_title('Training Accuracy Per Epoch')
        axs[1, 0].set_xlabel('Epoch')
        axs[1, 0].set_ylabel('Accuracy')
        axs[1, 0].grid(True)
        axs[1, 0].legend()

        # Testing accuracy per epoch
        axs[1, 1].plot(pytorch_data['Epoch'], pytorch_data['Test Accuracy'], marker='o', label='PyTorch')
        axs[1, 1].plot(tensorflow_data['Epoch'], tensorflow_data['Test Accuracy'], marker='o', label='TensorFlow')
        axs[1, 1].plot(jax_data['Epoch'], jax_data['Test Accuracy'], marker='o', label='JAX')
        axs[1, 1].set_title('Testing Accuracy Per Epoch')
        axs[1, 1].set_xlabel('Epoch')
        axs[1, 1].set_ylabel('Accuracy')
        axs[1, 1].grid(True)
        axs[1, 1].legend()

        # Adjust layout and save the plot
        plt.tight_layout()
        plt.savefig(output_plot_path)
        print(f"Comparison plots saved to {output_plot_path}")

    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
pytorch_csv = "./results/pt_metrics.csv"  # Path to PyTorch metrics CSV
tensorflow_csv = "./results/tf_metrics.csv"  # Path to TensorFlow metrics CSV
jax_csv = "./results/jax_metrics.csv"  # Path to JAX metrics CSV
output_plot_path = "./results/plots/framework_comparison_metrics.png"  # Path to save the output plot
generate_comparison_plots(pytorch_csv, tensorflow_csv, jax_csv, output_plot_path)
