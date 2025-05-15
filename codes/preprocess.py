# preprocess.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns   # type: ignore

def load_data(file_path, sep='\t', index_col=0):
    try:
        print(f"Loading {file_path}...")
        data = pd.read_csv(file_path, sep=sep, index_col=index_col)
        print(f"Successfully loaded: {data.shape[0]} rows × {data.shape[1]} columns")
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def inspect_dataset(data, name):
    print(f"\n==== Inspecting {name} dataset ====")
    print(f"Dimensions: {data.shape[0]} rows × {data.shape[1]} columns")

    missing = data.isna().sum().sum()
    missing_pct = 100 * missing / (data.shape[0] * data.shape[1])
    print(f"Missing values: {missing} ({missing_pct:.2f}%)")
    print(f"Data types:\n{data.dtypes.value_counts()}")

    if data.select_dtypes(include=[np.number]).shape[1] > 0:
        numeric_data = data.select_dtypes(include=[np.number])
        print(f"Value range: {numeric_data.min().min()} to {numeric_data.max().max()}")

    print("\nData preview:")
    print(data.iloc[:5, :5])
    
    return {
        'dimensions': data.shape,
        'missing': missing,
        'missing_pct': missing_pct
    }

def visualize_distributions(data, name, output_dir="results/figures/distributions"):
    os.makedirs(output_dir, exist_ok=True)
    
    if name == "cnv":
        # Histogram of values for CNV
        values = data.values.flatten()
        values = values[~np.isnan(values)]
        plt.figure(figsize=(10, 6))
        sns.histplot(values, bins=5, kde=False)
        plt.title(f'Distribution of {name.upper()} Values', fontsize=14)
        plt.xlabel('CNV Value (-2, -1, 0, 1, 2)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.xticks([-2, -1, 0, 1, 2])
        plt.savefig(f"{output_dir}/{name}_distribution.png")
        plt.close()
        
    elif name == "expression":
        # Boxplot of gene expression for genes
        sample_genes = np.random.choice(data.index, min(1000, len(data.index)), replace=False)
        sample_data = data.loc[sample_genes]
        
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=sample_data.T)
        plt.title(f'{name.capitalize()} Distribution Across Samples', fontsize=14)
        plt.xlabel('Genes', fontsize=12)
        plt.ylabel('Expression Value (log2)', fontsize=12)
        plt.xticks([]) 
        plt.savefig(f"{output_dir}/{name}_boxplot.png")
        plt.close()
        
        # Sensity plot of mean expression
        plt.figure(figsize=(10, 6))
        sns.kdeplot(sample_data.mean(axis=1), fill=True)
        plt.title(f'Mean {name.capitalize()} Distribution', fontsize=14)
        plt.xlabel('Mean Expression Value (log2)', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.savefig(f"{output_dir}/{name}_density.png")
        plt.close()
        
    elif name == "rppa":
        # Boxplot of protein expression for proteins
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=data.T)
        plt.title(f'{name.upper()} Protein Expression Distribution', fontsize=14)
        plt.xlabel('Proteins', fontsize=12)
        plt.ylabel('Protein Expression Value', fontsize=12)
        plt.xticks([]) 
        plt.savefig(f"{output_dir}/{name}_boxplot.png")
        plt.close()

if __name__ == "__main__":
    print("Part 2: Data Inspection and Quality Control")

    file_paths = {
        'cnv': "data/raw/brca_cnv_gistic2_thresholded.tsv",
        'expression': "data/raw/brca_expression_rnaseq.tsv",
        'rppa': "data/raw/brca_rppa.tsv",
        'clinical': "data/raw/brca_clinical.txt",
        'paradigm': "data/raw/brca_paradigm.tsv"
    }

    datasets = {}
    for name, path in file_paths.items():
        if name == 'clinical':
            datasets[name] = load_data(path, index_col=None)
        else:
            datasets[name] = load_data(path)

    inspection_results = {}
    for name, data in datasets.items():
        if data is not None:
            inspection_results[name] = inspect_dataset(data, name)

    for name, data in datasets.items():
        if data is not None and name != 'clinical' and name != 'paradigm':
            visualize_distributions(data, name)
    
    print("\nData inspection completed successfully!")