import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Enable LaTeX formatting in Matplotlib

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'  # Use serif fonts
plt.rcParams['text.latex.preamble'] = r'\usepackage{times}'  # Use Times New Roman

# Re-load the data
file_path = '/pfs/work7/workspace/scratch/cc7738-subgraph_train/TAPE/results/latest_mix/arxiv_2023.csv'
data = pd.read_csv(file_path)

# Function to extract the mean value from the string "mean ± std"
def extract_mean(value):
    try:
        mean_value = float(value.split(' ± ')[0])
    except:
        mean_value = None
    return mean_value

# Apply the function to relevant columns
for col in data.columns[1:]:  # Skip the 'Metric' column
    data[col] = data[col].apply(extract_mean)

# Define the groups (including all available models)
group_labels = {
    'NCNC': [r"\textbf{llama-ncnc}", r"\textbf{e5-ncnc}", r"\textbf{minilm-ncnc}", r"\textbf{bert-ncnc}"],
    'Buddy': [r"\textbf{llama-buddy}", r"\textbf{e5-buddy}", r"\textbf{minilm-buddy}", r"\textbf{bert-buddy}"],
        'NCN': [r"\textbf{llama-ncn}", r"\textbf{e5-ncn}", r"\textbf{minilm-ncn}", r"\textbf{bert-ncn}"],
    'HLGNN': [r"\textbf{llama-hlgnn}", r"\textbf{e5-hlgnn}", r"\textbf{minilm-hlgnn}", r"\textbf{bert-hlgnn}"],
    # 'NeoGNN': [r"\textbf{llama-neognn}", r"\textbf{e5-neognn}", r"\textbf{minilm-neognn}", r"\textbf{bert-neognn}"]
}

# Define bar width
bar_width = 3.0  # Increased bar width for wider bins

# Calculate group start indices dynamically
def calculate_group_start_indices(group_labels, bar_width, group_spacing_factor=7):
    current_position = 0
    group_start_indices = {}
    
    for group, models in group_labels.items():
        group_start_indices[group] = np.arange(len(models)) * (bar_width * 3) + current_position
        current_position = group_start_indices[group][-1] + bar_width * group_spacing_factor
    
    return group_start_indices

# Calculate dynamic group start indices
group_start_indices = calculate_group_start_indices(group_labels, bar_width)

# Prepare the data for plotting
mrr_data = []
hits50_data = []
auc_data = []
labels = []
x_positions = []

# Populate data for plotting
for group, models in group_labels.items():
    for i, model in enumerate(models):
        mrr_value = data.loc[data['Metric'] == model.split('{')[1].split('}')[0] + '-arxiv_2023', 'MRR'].values[0] if not data.loc[data['Metric'] == model.split('{')[1].split('}')[0] + '-arxiv_2023', 'MRR'].empty else 0
        hits50_value = data.loc[data['Metric'] == model.split('{')[1].split('}')[0] + '-arxiv_2023', 'Hits@50'].values[0] if not data.loc[data['Metric'] == model.split('{')[1].split('}')[0] + '-arxiv_2023', 'Hits@50'].empty else 0
        auc_value = data.loc[data['Metric'] == model.split('{')[1].split('}')[0] + '-arxiv_2023', 'AUC'].values[0] if not data.loc[data['Metric'] == model.split('{')[1].split('}')[0] + '-arxiv_2023', 'AUC'].empty else 0
        
        mrr_data.append(mrr_value)
        hits50_data.append(hits50_value)
        auc_data.append(auc_value)

        # labels.append(f"{model}")
        x_positions.append(group_start_indices[group][i])

labels = [r"\textbf{Llama}", r"\textbf{e5}", r"\textbf{Minilm}", r"\textbf{Bert}"] *4

plt.figure(figsize=(70, 48))  
plt.bar(x_positions, mrr_data, bar_width, label=r'\textbf{MRR}')
plt.bar([x + bar_width for x in x_positions], hits50_data, bar_width, label=r'\textbf{Hits@50}')
plt.bar([x + 2 * bar_width for x in x_positions], auc_data, bar_width, label=r'\textbf{AUC}')

fs = 132
for group_name, indices in group_start_indices.items():
    plt.text(np.mean(indices) + bar_width, np.mean(mrr_data + hits50_data + auc_data) * 0.75, f"{group_name}", 
             fontsize=fs, ha='center', va='bottom', fontweight='bold')

# plt.xlabel(r'\textbf{Model}', fontsize=fs)
plt.ylabel(r'\textbf{MRR, Hits@50, and AUC (\%)}', fontsize=fs)
plt.title(r'\textbf{Metrics Across Different Model Groups}', fontsize=164)
plt.xticks([x + bar_width for x in x_positions], labels, rotation=45, ha="right", fontsize=fs)
plt.yticks(fontsize=fs)
plt.legend(fontsize=fs, framealpha=0.4)  # Added transparency to the legend
plt.grid(True)

plt.savefig('plot.pdf')
plt.show()