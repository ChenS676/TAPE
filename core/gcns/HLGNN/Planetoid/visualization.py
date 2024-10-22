import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.image as mpimg
from matplotlib.colors import PowerNorm
import os
import re

def visualization(beta_values, data_name, dataset):
    # # Load beta values
    beta_values = []
    # data_name = "Pubmed"
    with open(f'metrics_and_weights/arxiv_2023_beta/beta_values{data_name.lower()}.txt', 'r') as f:
        for line in f:
            if 'hl_gnn_planetoid' not in line:
                # print(line)
                epoch, layer, value = line.strip().split('\t')
                # new_line = line.strip().split(' ')
                # epoch, layer, value = [element for element in new_line if element]
                beta_values.append((int(epoch), int(layer), float(value)))

    # Convert to numpy array for easier manipulation
    beta_values = np.array(beta_values)

    # Plot histograms for the final epoch
    final_epoch = max(beta_values[:, 0])
    final_beta_values = beta_values[beta_values[:, 0] == final_epoch]

    plt.figure(figsize=(12, 6))

    norm = PowerNorm(gamma=0.2, vmin=final_beta_values[:, 2].min(), vmax=final_beta_values[:, 2].max())
    colors = cm.Blues(norm(final_beta_values[:, 2]))

    plt.bar(final_beta_values[:, 1], final_beta_values[:, 2], color=colors)
    plt.xlabel('/th-layer')
    plt.ylabel(r'$\beta^{(l)}$')
    plt.title(f'Beta values at layer for {dataset} with embeddings from {data_name}')

    # Save the figure as a PNG file
    plt.savefig(f'beta_values_{data_name.lower()}_{dataset}.png')


def visualization_epochs(beta_values, data_name):
    beta_values = np.array(beta_values)
    unique_epochs = np.unique(beta_values[:, 0])

    for epoch in unique_epochs:
        current_epoch_beta_values = beta_values[beta_values[:, 0] == epoch]

        plt.figure(figsize=(12, 6))

        norm = PowerNorm(gamma=0.2, vmin=current_epoch_beta_values[:, 2].min(), vmax=current_epoch_beta_values[:, 2].max())
        colors = cm.Blues(norm(current_epoch_beta_values[:, 2]))

        plt.bar(current_epoch_beta_values[:, 1], current_epoch_beta_values[:, 2], color=colors)
        plt.xlabel('/th-layer')
        plt.ylabel(r'$\beta^{(l)}$')
        plt.title(f'Beta values at layer for {data_name} - Epoch {epoch}')

        # Save the figure as a PNG file
        plt.savefig(f'hl_gnn_planetoid/graph_results/beta_values_{data_name.lower()}_epoch_{epoch}.png')
        
def visualization_geom_fig(beta_values, graph_type, m, n, homo, hetero):
    # Convert to numpy array for easier manipulation
    beta_values = np.array(beta_values)

    # Plot histograms for the final epoch
    print(beta_values)
    final_epoch = max(beta_values[:, 0])
    final_beta_values = beta_values[beta_values[:, 0] == final_epoch]

    plt.figure(figsize=(20, 6))

    # Normalize the beta values for color mapping
    norm = PowerNorm(gamma=0.2, vmin=final_beta_values[:, 2].min(), vmax=final_beta_values[:, 2].max())
    colors = cm.Blues(norm(final_beta_values[:, 2]))

    # Create the bar plot
    plt.bar(final_beta_values[:, 1], final_beta_values[:, 2], color=colors)

    plt.xlabel('/th-layer')
    plt.ylabel(r'$\beta^{(l)}$')
    plt.title(f'Beta values at layer for {graph_type.upper()} with m={m} and n={n}')

    # Save the figure as a PNG file
    if homo == True:
        plt.savefig(f'graph_results/homo_beta_values_{graph_type.lower()}_{m}_{n}.png')
    elif hetero == True:
        plt.savefig(f'graph_results/hetero_beta_values_{graph_type.lower()}_{m}_{n}.png')

def matrix_visualization():
    number = 50
    first_part = [[f'beta_values_grid_{number}_{i}' for i in range(10, 35, 5)],
                 [f'beta_values_hexagonal_{number}_{i}' for i in range(10, 35, 5)],
                 [f'beta_values_kagome_{number}_{i}' for i in range(10, 35, 5)],
                 [f'beta_values_triangle_{number}_{i}' for i in range(10, 35, 5)],
                ]
    
    second_part = [[f'beta_values_grid_{number}_{i}' for i in range(35, 55, 5)],
                 [f'beta_values_hexagonal_{number}_{i}' for i in range(35, 55, 5)],
                 [f'beta_values_kagome_{number}_{i}' for i in range(35, 55, 5)],
                 [f'beta_values_triangle_{number}_{i}' for i in range(35, 55, 5)],
                ]

    for lst_names in [first_part, second_part]:
        if lst_names == first_part:
            num_cols = 5
            name = 'first_part'
        else:
            num_cols = 4
            name = 'second_part'
        # Flatten the list of lists
        flattened_lst_names = [item for sublist in lst_names for item in sublist]
        
        # Define the path to the directory containing the images
        image_dir = 'graph_results/'

        # Number of rows and columns for the plot
        num_rows = 4

        # Create a figure with the specified number of subplots
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 15))

        # Flatten the axes array for easy iteration
        axes = axes.flatten()

        # Loop through the sorted graph names and plot each image
        for i, graph_name in enumerate(flattened_lst_names):
            # Construct the full path to the image
            img_path = f'{image_dir}/{graph_name}.png'
            
            # Load the image
            img = mpimg.imread(img_path)
            gr_name = graph_name.split('_')
            # Plot the image in the corresponding subplot
            axes[i].imshow(img)
            axes[i].set_title(f'{gr_name[2][0].upper() + gr_name[2][1:]} with m={gr_name[3]} n={gr_name[4]}' )
            axes[i].axis('off')  # Turn off axis

        # Adjust layout
        plt.tight_layout()
        # Save the figure as a PNG file
        plt.savefig(f'graph_results/matrix_graph_{number}_{name}.png')
    
def visualization_beta(results):
    fig, ax1 = plt.subplots(figsize=(10, 10))

    # Primary y-axis
    ax1.set_xlabel('Layer (l)')
    ax1.set_ylabel('Beta (β)')

    for alpha in results:
        l_vals = [item[2] for item in results[alpha]]
        beta_vals = [item[1] for item in results[alpha]]
        ax1.plot(beta_vals, l_vals, marker='o', markersize=5, label=f'α={alpha:.2f}')

    ax1.tick_params(axis='y')
    ax1.legend(title=r'$\alpha$', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Title
    plt.title(r'$KI, hexagonal, heterophily$', fontsize=16)
    fig.tight_layout()  # Adjust layout to prevent overlap
    plt.savefig(f'graph_results/z_beta.png')

def visualization_zero_weight(file_path, num_layer):
    
    with open(file_path, 'r') as file:
        content = file.read()
    lines = content.strip().split('\n')
    values = []
    for line in lines[1:]:
        if 'hl_gnn_planetoid' not in line:
            epoch, layer, value = line.split()
            epoch = int(epoch)
            value = float(value)
            layer = int(layer)
            if layer == num_layer:
                values.append(value)
    print(values)
    return np.mean(values)

# if __name__=="__main__":
#     # matrix_visualization()
#     # List of files
#     files = ['metrics_and_weights/beta_valuesbert.txt', 'metrics_and_weights/beta_valuese5.txt', 
#             'metrics_and_weights/beta_valuesminilm.txt','metrics_and_weights/beta_valuesllama.txt']

#     for num_layer in range(0, 21):
#         # Dictionary to hold method names and their corresponding values
#         data = {}
#         names = ['BERT', 'E5-Large', 'MiniLM', 'LLAMA']
#         for file, method_name in zip(files, names):
#             values = visualization_zero_weight(file, num_layer)
#             data[method_name] = values

#         # Plotting
#         plt.figure(figsize=(12, 6))

#         for method_name, values in data.items():
#             plt.plot(method_name, values, marker='o', label=method_name)

#         plt.xlabel('Embbeding name')
#         plt.ylabel('Value')
#         plt.title(rf'Values for $\beta^{({num_layer})}$')
#         plt.legend()
#         plt.grid(True)
#         plt.savefig(f'graph_results/pubmed_{num_layer}_weight.png')

# import matplotlib.pyplot as plt
# import numpy as np
# import matplotlib.cm as cm

def extract_layer_values(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    blocks = content.strip().split('\n\n')
    
    values = []
    for block in blocks:
        lines = block.strip().split('\n')
        for line in lines[1:]:
            # epoch, layer, value = line.split()
            if 'hl_gnn_planetoid' not in line:
                new_line = line.strip().split('\t')
                # print(line)
                # print(new_line)
                epoch, layer, value = [element for element in new_line if element]
                epoch = int(epoch)
                value = float(value)
                layer = int(layer)
                values.append((layer,value))
        # For this task, we only process the first block
        break
    return values

def plot_layer_values(values, data_name):
    plt.figure(figsize=(12, 6))
    
    normalized_index = np.arange(len(values)) / float(len(values))
    colors = cm.Blues(normalized_index)
    
    plt.bar(values[0], values[1], color=colors)
    plt.xlabel('Index')
    plt.ylabel('layer')
    # plt.title(rf'Values for $\beta^{{{layer_num}}}$ - {data_name}')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'beta_values_layer_{data_name}.png')
    # plt.show()

if __name__=="__main__":
    # Path to your file
    dataset = 'Arxiv_2023'
    for name in ['bert', 'e5', 'llama', 'minilm']:
    # layer_num = 0  # Layer you are interested in
        visualization([], name, dataset)
