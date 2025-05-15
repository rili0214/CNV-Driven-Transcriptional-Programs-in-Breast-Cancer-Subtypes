# id_map.py
import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns   # type: ignore

def map_gene_ids(expr_data):
    print("Mapping RNA-seq gene identifiers to standard gene symbols...")

    gene_map = {}

    for gene_id in expr_data.index:
        if '|' in gene_id:
            symbol = gene_id.split('|')[0]
            if symbol != '?' and len(symbol) > 0:
                gene_map[gene_id] = symbol
        else:
            gene_map[gene_id] = gene_id

    expr_data_mapped = expr_data.copy()
    expr_data_mapped.index = expr_data_mapped.index.map(lambda x: gene_map.get(x, x))

    if len(expr_data_mapped.index) != len(expr_data_mapped.index.unique()):
        print("Found duplicate gene symbols after mapping. Taking mean of duplicates.")
        expr_data_mapped = expr_data_mapped.groupby(level=0).mean()
    
    print(f"Mapped {len(gene_map)} gene IDs. Final gene count: {expr_data_mapped.shape[0]}")
    return expr_data_mapped, gene_map

def identify_biomarkers(expr_data, rppa_data):
    print("\nIdentifying key breast cancer biomarkers...")
    
    # ESR1: https://breast-cancer-research.biomedcentral.com/articles/10.1186/s13058-021-01462-3
    # PGR: https://pmc.ncbi.nlm.nih.gov/articles/PMC7152560/
    # ERBB2: https://www.nature.com/articles/s41419-022-05117-9
    # MKI67: https://pmc.ncbi.nlm.nih.gov/articles/PMC8430879/
    # EGFR: https://pmc.ncbi.nlm.nih.gov/articles/PMC3832208/
    # KRT5: https://pubmed.ncbi.nlm.nih.gov/37155625/ 
    # KRT14: https://pmc.ncbi.nlm.nih.gov/articles/PMC7642004/ 
    # FOXA1: https://pmc.ncbi.nlm.nih.gov/articles/PMC8533709/
    # GATA3: https://pmc.ncbi.nlm.nih.gov/articles/PMC4758516/ 
    # ER: https://pmc.ncbi.nlm.nih.gov/articles/PMC7922594/ 
    # PR: https://pubmed.ncbi.nlm.nih.gov/35006257/
    # HER2: https://pubmed.ncbi.nlm.nih.gov/35006257/
    # Ki67: https://pubmed.ncbi.nlm.nih.gov/36730064/ 
    # p53: https://pubmed.ncbi.nlm.nih.gov/29564741/ 
    key_markers = {
        'RNA': ['ESR1', 'PGR', 'ERBB2', 'MKI67', 'EGFR', 'KRT5', 'KRT14', 'FOXA1', 'GATA3'],
        'Protein': ['ER', 'PR', 'HER2', 'EGFR', 'Ki67', 'p53']
    }
    
    rna_present = [gene for gene in key_markers['RNA'] if gene in expr_data.index]
    rna_missing = [gene for gene in key_markers['RNA'] if gene not in expr_data.index]
    print(f"RNA biomarkers present: {len(rna_present)}/{len(key_markers['RNA'])}")
    print(f"Present: {', '.join(rna_present)}")
    print(f"Missing: {', '.join(rna_missing)}")
    
    protein_matches = {}
    for marker in key_markers['Protein']:
        pattern = re.compile(f".*{marker}.*", re.IGNORECASE)
        matches = [protein for protein in rppa_data.index if pattern.match(protein)]
        protein_matches[marker] = matches
    
    print("\nProtein biomarkers identified in RPPA data:")
    for marker, matches in protein_matches.items():
        if matches:
            print(f"- {marker}: {', '.join(matches)}")
        else:
            print(f"- {marker}: No matches found")
    
    return {
        'rna_present': rna_present,
        'protein_matches': protein_matches
    }

def visualize_biomarkers(expr_data, rppa_data, biomarkers, output_dir="results/figures/biomarkers"):
    os.makedirs(output_dir, exist_ok=True)
    
    # RNA biomarker distributions
    if biomarkers['rna_present']:
        rna_data = expr_data.loc[biomarkers['rna_present']]
        
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=rna_data.T)
        plt.title('RNA Biomarker Expression Distribution', fontsize=14)
        plt.xlabel('Gene', fontsize=12)
        plt.ylabel('Expression (log2)', fontsize=12)
        plt.xticks(range(len(rna_data.index)), rna_data.index, rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/rna_biomarkers_boxplot.png")
        plt.close()
    
    # Protein biomarker distributions
    protein_markers = []
    for marker, matches in biomarkers['protein_matches'].items():
        if matches:
            protein_markers.extend(matches)
    
    if protein_markers:
        protein_data = rppa_data.loc[protein_markers]
        
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=protein_data.T)
        plt.title('Protein Biomarker Expression Distribution', fontsize=14)
        plt.xlabel('Protein', fontsize=12)
        plt.ylabel('RPPA Expression', fontsize=12)
        plt.xticks(range(len(protein_data.index)), protein_data.index, rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/protein_biomarkers_boxplot.png")
        plt.close()

def save_biomarker_data(expr_data, rppa_data, biomarkers):
    print("\nSaving biomarker data for subtyping...")

    if biomarkers['rna_present']:
        rna_biomarker_data = expr_data.loc[biomarkers['rna_present']]
        rna_biomarker_data.to_csv("data/processed/rna_biomarkers.tsv", sep='\t')
        print(f"Saved RNA biomarker data with shape: {rna_biomarker_data.shape}")

    protein_markers = []
    for marker, matches in biomarkers['protein_matches'].items():
        if matches:
            protein_markers.extend(matches)
    
    if protein_markers:
        protein_biomarker_data = rppa_data.loc[protein_markers]
        protein_biomarker_data.to_csv("data/processed/protein_biomarkers.tsv", sep='\t')
        print(f"Saved protein biomarker data with shape: {protein_biomarker_data.shape}")

    pd.DataFrame({
        'RNA_marker': biomarkers['rna_present'],
        'Present': True
    }).to_csv("data/processed/rna_marker_mapping.tsv", sep='\t', index=False)
    
    protein_mapping_data = []
    for marker, matches in biomarkers['protein_matches'].items():
        for match in matches:
            protein_mapping_data.append({
                'Protein_marker': marker,
                'RPPA_antibody': match
            })
    
    if protein_mapping_data:
        pd.DataFrame(protein_mapping_data).to_csv(
            "data/processed/protein_marker_mapping.tsv", sep='\t', index=False)

if __name__ == "__main__":
    print("Part 4: Gene and Protein ID Mapping")

    expr_data = pd.read_csv("data/processed/matched_expression.tsv", sep='\t', index_col=0)
    rppa_data = pd.read_csv("data/processed/matched_rppa.tsv", sep='\t', index_col=0)

    expr_data_mapped, gene_map = map_gene_ids(expr_data)
    expr_data_mapped.to_csv("data/processed/expression_mapped.tsv", sep='\t')

    biomarkers = identify_biomarkers(expr_data_mapped, rppa_data)
    visualize_biomarkers(expr_data_mapped, rppa_data, biomarkers)
    save_biomarker_data(expr_data_mapped, rppa_data, biomarkers)
    
    print("\nGene and protein ID mapping completed successfully!")