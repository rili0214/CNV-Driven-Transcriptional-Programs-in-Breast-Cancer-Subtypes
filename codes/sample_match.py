# sample_match.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib_venn import venn3   # type: ignore
import os

def extract_sample_ids(datasets):
    sample_ids = {}
    
    for name, data in datasets.items():
        if name == 'clinical':
            if 'sample' in data.columns:
                sample_ids[name] = data['sample'].tolist()
            else:
                for col in data.columns:
                    if any('TCGA' in str(x) for x in data[col].values):
                        sample_ids[name] = data[col].tolist()
                        break
        else:
            sample_ids[name] = data.columns.tolist()
    
    return sample_ids

def find_common_samples(sample_ids):
    all_common = set.intersection(*[set(ids) for ids in sample_ids.values()])

    common = {}
    for name1, ids1 in sample_ids.items():
        for name2, ids2 in sample_ids.items():
            if name1 < name2:
                common[f"{name1}_{name2}"] = set(ids1).intersection(set(ids2))
    
    return all_common, common

def visualize_sample_overlap(sample_ids, output_dir="results/figures"):
    os.makedirs(output_dir, exist_ok=True)
    
    # Venn diagram for CNV, expression, RPPA datasets
    plt.figure(figsize=(10, 10))
    venn3([set(sample_ids['cnv']), set(sample_ids['expression']), set(sample_ids['rppa'])],
          ('CNV', 'Expression', 'RPPA'))
    plt.title('Sample Overlap Between Key Datasets', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/sample_overlap_venn.png")
    plt.close()
    
    # Bar chart for sample counts
    plt.figure(figsize=(10, 6))
    counts = {name: len(ids) for name, ids in sample_ids.items()}
    plt.bar(counts.keys(), counts.values())
    plt.title('Number of Samples per Dataset', fontsize=14)
    plt.ylabel('Sample Count', fontsize=12)
    plt.xticks(rotation=45)
    for i, (name, count) in enumerate(counts.items()):
        plt.text(i, count + 5, str(count), ha='center')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/sample_counts.png")
    plt.close()

def create_matched_datasets(datasets, common_samples):
    matched_datasets = {}
    
    for name, data in datasets.items():
        if name == 'clinical':
            if 'sample' in data.columns:
                matched_datasets[name] = data[data['sample'].isin(common_samples)]
            else:
                for col in data.columns:
                    if any('TCGA' in str(x) for x in data[col].values):
                        matched_datasets[name] = data[data[col].isin(common_samples)]
                        break
        else:
            matched_datasets[name] = data[[col for col in data.columns if col in common_samples]]
    
    return matched_datasets

if __name__ == "__main__":
    print("Part 3: Sample Matching and ID Standardization")

    datasets = {}
    for name, path in {
        'cnv': "data/raw/brca_cnv_gistic2_thresholded.tsv",
        'expression': "data/raw/brca_expression_rnaseq.tsv",
        'rppa': "data/raw/brca_rppa.tsv",
        'clinical': "data/raw/brca_clinical.txt"
    }.items():
        if name == 'clinical':
            datasets[name] = pd.read_csv(path, sep='\t')
        else:
            datasets[name] = pd.read_csv(path, sep='\t', index_col=0)
    
    sample_ids = extract_sample_ids(datasets)
    print("\nSample counts by dataset:")
    for name, ids in sample_ids.items():
        print(f"- {name}: {len(ids)} samples")

    all_common, pairwise_common = find_common_samples(sample_ids)
    print(f"\nSamples common to all datasets: {len(all_common)}")
    for pair, common in pairwise_common.items():
        print(f"- Samples common to {pair}: {len(common)}")

    visualize_sample_overlap(sample_ids)

    matched_datasets = create_matched_datasets(datasets, all_common)

    print("\nSaving matched datasets...")
    for name, data in matched_datasets.items():
        if name == 'clinical':
            data.to_csv(f"data/processed/matched_{name}.tsv", sep='\t', index=False)
        else:
            data.to_csv(f"data/processed/matched_{name}.tsv", sep='\t')
        print(f"- Saved {name} dataset with {data.shape} dimensions")
    
    print("\nSample matching completed successfully!")
