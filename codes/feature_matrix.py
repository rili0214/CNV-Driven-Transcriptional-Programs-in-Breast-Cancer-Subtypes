import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
import warnings

warnings.filterwarnings('ignore')

KEY_MARKER_GENES = {
    'ESR1': 'Luminal', 'PGR': 'Luminal', 'FOXA1': 'Luminal', 'GATA3': 'Luminal',
    'XBP1': 'Luminal', 'TFF1': 'Luminal', 'ERBB2': 'HER2', 'GRB7': 'HER2',
    'STARD3': 'HER2', 'MKI67': 'Proliferation', 'PCNA': 'Proliferation',
    'AURKA': 'Proliferation', 'BIRC5': 'Proliferation', 'CCNB1': 'Proliferation',
    'MYBL2': 'Proliferation', 'KRT5': 'Basal', 'KRT14': 'Basal', 'KRT17': 'Basal',
    'FOXC1': 'Basal', 'MYC': 'Basal', 'CDH3': 'Basal', 'EGFR': 'Basal',
    'KRT18': 'Normal', 'BCL2': 'Normal', 'CCND1': 'Luminal', 'FGFR1': 'Luminal B',
    'TP53': 'Various', 'PIK3CA': 'Various', 'PTEN': 'Various', 'BRCA1': 'Basal',
    'CDK4': 'Luminal B', 'RB1': 'Various', 'CDKN2A': 'Basal'
}

KEY_CNV_REGIONS = {
    'ERBB2': 'HER2', 'CCND1': 'Luminal', 'MYC': 'Basal/Luminal B',
    'FGFR1': 'Luminal B', 'EGFR': 'Basal', 'PTEN': 'Various',
    'RB1': 'Various', 'BRCA1': 'Basal', 'PIK3CA': 'Various', 'CDKN2A': 'Basal'
}

KEY_PROTEIN_MARKERS = [
    'ER-alpha', 'ER', 'ESR1', 'PR', 'PGR', 'Progesterone', 'HER2', 'ERBB2',
    'HER2_pY1248', 'EGFR', 'EGFR_pY1068', 'EGFR_pY1173', 'Ki67', 'MKI67',
    'Cyclin_D1', 'CCND1', 'GATA3', 'PTEN', 'p53', 'TP53', 'Akt', 'Akt_pS473',
    'Akt_pT308', 'mTOR', 'mTOR_pS2448', 'INPP4B', 'Caveolin-1', 'E-cadherin',
    'CDH1', 'Claudin-7', 'Rb', 'Rb_pS807_S811', 'Bcl-2', 'BCL2', 'Collagen_VI',
    'beta-Catenin'
]

def load_data_with_error_handling():
    datasets = {}
    try:
        files = {
            'expr': "data/processed/expression_mapped.tsv",
            'rppa': "data/processed/matched_rppa.tsv",
            'cnv': "data/processed/matched_cnv.tsv",
            'clinical': "data/processed/matched_clinical.tsv",
            'rna_markers': "data/processed/rna_marker_mapping.tsv",
            'protein_markers': "data/processed/protein_marker_mapping.tsv"
        }
        
        for key, file_path in files.items():
            if os.path.exists(file_path):
                if key in ['expr', 'rppa', 'cnv']:
                    datasets[key] = pd.read_csv(file_path, sep='\t', index_col=0)
                else:
                    datasets[key] = pd.read_csv(file_path, sep='\t')
                print(f"Loaded {key} data: {datasets[key].shape}")
            else:
                print(f"Warning: File not found at {file_path}")
    except Exception as e:
        print(f"Error loading data: {str(e)}")
    return datasets

def select_key_genes(expr_data, key_genes=KEY_MARKER_GENES):
    available_markers = []
    marker_categories = {}
    
    for gene, category in key_genes.items():
        if gene in expr_data.index:
            available_markers.append(gene)
            marker_categories[gene] = category
        else:
            matches = [g for g in expr_data.index if g.upper() == gene.upper()]
            if matches:
                available_markers.append(matches[0])
                marker_categories[matches[0]] = category
                
    print(f"Found {len(available_markers)}/{len(key_genes)} key marker genes")
    
    if available_markers:
        return expr_data.loc[available_markers], marker_categories
    else:
        print("Warning: Using most variable genes instead")
        top_var_genes = expr_data.var(axis=1).nlargest(30).index
        return expr_data.loc[top_var_genes], {}

def find_protein_markers(rppa_data, key_proteins=KEY_PROTEIN_MARKERS):
    available_proteins = []
    
    for protein in key_proteins:
        if protein in rppa_data.index:
            available_proteins.append(protein)
        else:
            matches = [p for p in rppa_data.index if protein.lower() in p.lower()]
            if matches:
                available_proteins.extend(matches)
                
    available_proteins = list(dict.fromkeys(available_proteins))
    print(f"Found {len(available_proteins)} key protein markers")
    
    if available_proteins:
        return rppa_data.loc[available_proteins]
    else:
        print("Warning: Using most variable proteins instead")
        top_var_proteins = rppa_data.var(axis=1).nlargest(20).index
        return rppa_data.loc[top_var_proteins]

def select_cnv_regions(cnv_data, key_regions=KEY_CNV_REGIONS):
    available_regions = []
    region_categories = {}
    
    for gene, category in key_regions.items():
        if gene in cnv_data.index:
            available_regions.append(gene)
            region_categories[gene] = category
            
    print(f"Found {len(available_regions)}/{len(key_regions)} key CNV regions")
    
    if available_regions:
        return cnv_data.loc[available_regions], region_categories
    else:
        print("Warning: Using most variable regions instead")
        top_var_regions = cnv_data.var(axis=1).nlargest(10).index
        return cnv_data.loc[top_var_regions], {}

def plot_feature_distributions(feature_data, title, category_mapping=None, output_dir="results/figures/features"):
    os.makedirs(output_dir, exist_ok=True)
    data_t = feature_data.T

    missing_values = data_t.isna().sum().sum()
    if missing_values > 0:
        print(f"Warning: Found {missing_values} missing values in {title} data")
        data_t_filled = data_t.fillna(data_t.median())
    else:
        data_t_filled = data_t

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_t_filled)
        
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data_scaled)
        
    pca_df = pd.DataFrame({
        'PC1': pca_result[:, 0],
        'PC2': pca_result[:, 1],
        'Sample': data_t.index
    })
        
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', alpha=0.7)
    plt.title(f'PCA of {title} Features')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.grid(alpha=0.3)
    plt.savefig(f"{output_dir}/{title.lower().replace(' ', '_')}_pca.png")
    plt.close()

    plt.figure(figsize=(12, 10))
    correlation = data_t_filled.corr(method='pearson')
    sns.heatmap(correlation, cmap='coolwarm', center=0, annot=False)
    plt.title(f'Correlation Matrix of {title} Features')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{title.lower().replace(' ', '_')}_correlation.png")
    plt.close()

    if category_mapping:
        feature_data_long = data_t_filled.melt(
            var_name='Feature', value_name='Value', ignore_index=False
        ).reset_index().rename(columns={'index': 'Sample'})
            
        feature_data_long['Category'] = feature_data_long['Feature'].map(
            lambda x: category_mapping.get(x, 'Other')
        )
            
        plt.figure(figsize=(14, 8))
        sns.boxplot(data=feature_data_long, x='Feature', y='Value', hue='Category', dodge=False)
        plt.xticks(rotation=90)
        plt.title(f'{title} Features by Category')
        plt.legend(title='Category')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{title.lower().replace(' ', '_')}_by_category.png")
        plt.close()

def create_enhanced_feature_matrix(datasets, use_variance_filtering=True):
    expr_data = datasets.get('expr')
    rppa_data = datasets.get('rppa')
    cnv_data = datasets.get('cnv')
    
    if not all([expr_data is not None, rppa_data is not None, cnv_data is not None]):
        print("Error: Missing required datasets")
        return None, None

    marker_expr, expr_categories = select_key_genes(expr_data)
    protein_markers = find_protein_markers(rppa_data)
    cnv_regions, cnv_categories = select_cnv_regions(cnv_data)

    common_samples = list(set(marker_expr.columns) & 
                         set(protein_markers.columns) & 
                         set(cnv_regions.columns))
    
    print(f"Found {len(common_samples)} common samples across all data types")
    
    if len(common_samples) < 10:
        print("Error: Too few common samples")
        return None, None

    marker_expr_t = marker_expr[common_samples].T
    protein_markers_t = protein_markers[common_samples].T
    cnv_regions_t = cnv_regions[common_samples].T

    for df_name, df in [("Expression", marker_expr_t), 
                        ("Protein", protein_markers_t),
                        ("CNV", cnv_regions_t)]:
        missing = df.isna().sum().sum()
        if missing > 0:
            print(f"  {df_name} data: {missing} missing values - imputing")
            df.fillna(df.median(), inplace=True)

    expr_scaler = RobustScaler()
    protein_scaler = RobustScaler()
    
    marker_expr_scaled = pd.DataFrame(
        expr_scaler.fit_transform(marker_expr_t),
        index=marker_expr_t.index,
        columns=marker_expr_t.columns
    )
    
    protein_markers_scaled = pd.DataFrame(
        protein_scaler.fit_transform(protein_markers_t),
        index=protein_markers_t.index,
        columns=protein_markers_t.columns
    )
    
    # Add prefixes to column names
    marker_expr_scaled.columns = ['RNA_' + col for col in marker_expr_scaled.columns]
    protein_markers_scaled.columns = ['Protein_' + col for col in protein_markers_scaled.columns]
    cnv_regions_t.columns = ['CNV_' + col for col in cnv_regions_t.columns]

    combined_features = pd.concat(
        [marker_expr_scaled, protein_markers_scaled, cnv_regions_t], 
        axis=1
    )
    
    missing_values = combined_features.isna().sum().sum()
    if missing_values > 0:
        print(f"Warning: {missing_values} missing values in combined matrix - imputing")
        combined_features = combined_features.fillna(combined_features.median())

    if use_variance_filtering:
        var_filter = VarianceThreshold(threshold=0.1)
        filtered_data = var_filter.fit_transform(combined_features)
        
        selected_features = [combined_features.columns[i] for i in range(combined_features.shape[1])
                            if var_filter.get_support()[i]]
        
        combined_features = pd.DataFrame(
            filtered_data,
            index=combined_features.index,
            columns=selected_features
        )
        
        print(f"Variance filtering reduced features from {var_filter.get_support().size} to {len(selected_features)}")

    feature_types = {}

    for gene, category in expr_categories.items():
        feature_types[f'RNA_{gene}'] = category
    
    for gene, category in cnv_categories.items():
        feature_types[f'CNV_{gene}'] = category

    for protein in protein_markers.index:
        gene_match = next((gene for gene in expr_categories if gene.upper() in protein.upper()), None)
        
        if gene_match:
            feature_types[f'Protein_{protein}'] = expr_categories[gene_match]
        else:
            feature_types[f'Protein_{protein}'] = 'Protein'
    
    return combined_features, feature_types

def save_feature_matrix(features, output_dir="data/processed"):
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, "integrated_features.tsv")
    features.to_csv(output_file, sep='\t')
    
    imputed_file = os.path.join(output_dir, "integrated_features_imputed.tsv")
    features.fillna(features.median()).to_csv(imputed_file, sep='\t')
    
    print(f"Saved feature matrices to {output_file} and {imputed_file}")
    return output_file, imputed_file

def analyze_feature_importance(features, feature_types, output_dir="results/figures/features"):
    os.makedirs(output_dir, exist_ok=True)
    
    feature_variance = features.var().sort_values(ascending=False)

    variance_df = pd.DataFrame({
        'Feature': feature_variance.index,
        'Variance': feature_variance.values
    })

    variance_df['Type'] = variance_df['Feature'].apply(lambda x: x.split('_')[0])
    variance_df['Category'] = variance_df['Feature'].apply(lambda x: feature_types.get(x, 'Other'))

    plt.figure(figsize=(14, 8))
    sns.barplot(data=variance_df.head(30), x='Feature', y='Variance', hue='Type')
    plt.xticks(rotation=90)
    plt.title('Top 30 Features by Variance')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_variance_by_type.png")
    plt.close()

    plt.figure(figsize=(14, 8))
    sns.barplot(data=variance_df.head(30), x='Feature', y='Variance', hue='Category')
    plt.xticks(rotation=90)
    plt.title('Top 30 Features by Variance and Category')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_variance_by_category.png")
    plt.close()

    feature_corr = features.corr()

    corr_pairs = []
    
    for i in range(len(feature_corr.columns)):
        for j in range(i+1, len(feature_corr.columns)):
            f1, f2 = feature_corr.columns[i], feature_corr.columns[j]
            corr_val = feature_corr.iloc[i, j]
            
            if abs(corr_val) > 0.5:
                corr_pairs.append({
                    'Feature1': f1, 'Feature2': f2,
                    'Correlation': corr_val, 'AbsCorrelation': abs(corr_val)
                })

    if corr_pairs:
        corr_df = pd.DataFrame(corr_pairs).sort_values('AbsCorrelation', ascending=False)
        
        plt.figure(figsize=(10, 10))
        for i, row in corr_df.head(10).iterrows():
            plt.scatter(
                features[row['Feature1']], features[row['Feature2']], alpha=0.7,
                label=f"{row['Feature1']} vs {row['Feature2']} (r={row['Correlation']:.2f})"
            )
        
        plt.xlabel('Feature 1 Value')
        plt.ylabel('Feature 2 Value')
        plt.title('Top 10 Feature Correlations')
        plt.legend(fontsize=8)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/top_feature_correlations.png")
        plt.close()
    
    return variance_df

def main():
    print("Creating integrated feature matrix for breast cancer subtyping...")

    datasets = load_data_with_error_handling()
    if not datasets:
        print("Error: Failed to load required datasets.")
        return
    
    # Extract and explore data from each source
    if 'expr' in datasets:
        marker_genes, gene_categories = select_key_genes(datasets['expr'])
        plot_feature_distributions(marker_genes, "RNA Marker", gene_categories)
    
    if 'rppa' in datasets:
        protein_data = find_protein_markers(datasets['rppa'])
        plot_feature_distributions(protein_data, "Protein Marker")
    
    if 'cnv' in datasets:
        cnv_regions, cnv_categories = select_cnv_regions(datasets['cnv'])
        plot_feature_distributions(cnv_regions, "CNV Region", cnv_categories)
    
    # Create integrated feature matrix
    features, feature_types = create_enhanced_feature_matrix(datasets)
    if features is None:
        print("Error: Failed to create feature matrix.")
        return
    
    # Analyze and save results
    variance_df = analyze_feature_importance(features, feature_types)
    output_file, imputed_file = save_feature_matrix(features)
    
    print(f"Feature matrix created with {features.shape[1]} features and {features.shape[0]} samples")
    
    return features, feature_types

if __name__ == "__main__":
    main()