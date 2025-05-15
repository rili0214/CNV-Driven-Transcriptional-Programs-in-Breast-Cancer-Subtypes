import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import fisher_exact, pearsonr, hypergeom
from statsmodels.stats.multitest import multipletests
import time
import yaml
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from matplotlib.patches import Patch

def safe_pearsonr(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan
    return pearsonr(x, y)

SUBTYPE_COLORS = {
    'Luminal A': '#1f77b4', 'Luminal B': '#17becf', 'Luminal B/HER2+': '#9467bd',
    'HER2-enriched': '#e377c2', 'Basal-like': '#d62728', 'Normal-like': '#2ca02c'
}

def load_config():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config

def load_processed_data():
    cnv_data = pd.read_csv("data/processed/matched_cnv.tsv", sep='\t', index_col=0)
    expr_data = pd.read_csv("data/processed/expression_mapped.tsv", sep='\t', index_col=0)
    sample_info = pd.read_csv("data/processed/clinical_with_subtypes.tsv", sep='\t')
    protein_data = pd.read_csv("data/processed/matched_rppa.tsv", sep='\t', index_col=0)
    gene_locations = pd.read_csv("data/processed/gene_locations.tsv", sep='\t', index_col=0)
    
    return {
        'cnv_data': cnv_data, 'expr_data': expr_data, 'sample_info': sample_info,
        'protein_data': protein_data, 'gene_locations': gene_locations, 'external_data': None
    }

def impute_missing_data(data_dict):
    config = load_config()
    if not config.get("impute_missing_values", True):
        return data_dict
    
    method = config.get("imputation_method", "knn")
    imputed_dict = {}
    
    cnv_data = data_dict['cnv_data']
    cnv_missing = cnv_data.isna().sum().sum()
    
    if cnv_missing > 0:
        if method == "knn":
            imputer = KNNImputer(n_neighbors=5)
            imputed_values = imputer.fit_transform(cnv_data.T).T 
            imputed_dict['cnv_data'] = pd.DataFrame(imputed_values, index=cnv_data.index, columns=cnv_data.columns)
        elif method == "mice":
            imputer = IterativeImputer(max_iter=10, random_state=42)
            imputed_values = imputer.fit_transform(cnv_data.T).T
            imputed_dict['cnv_data'] = pd.DataFrame(imputed_values, index=cnv_data.index, columns=cnv_data.columns)
    else:
        imputed_dict['cnv_data'] = cnv_data

    expr_data = data_dict['expr_data']
    expr_missing = expr_data.isna().sum().sum()
    
    if expr_missing > 0:
        if method == "knn":
            imputer = KNNImputer(n_neighbors=5)
            imputed_values = imputer.fit_transform(expr_data.T).T
            imputed_dict['expr_data'] = pd.DataFrame(imputed_values, index=expr_data.index, columns=expr_data.columns)
        elif method == "mice":
            imputer = IterativeImputer(max_iter=10, random_state=42)
            imputed_values = imputer.fit_transform(expr_data.T).T
            imputed_dict['expr_data'] = pd.DataFrame(imputed_values, index=expr_data.index, columns=expr_data.columns)
    else:
        imputed_dict['expr_data'] = expr_data
    
    for key in ['sample_info', 'protein_data', 'external_data', 'gene_locations']:
        imputed_dict[key] = data_dict[key]
    
    return imputed_dict

def load_driver_gene_databases():
    known_cancer_genes = {
        'ERBB2': {'role': 'Oncogene', 'source': 'Manual', 'evidence': 'Strong'},
        'MYC': {'role': 'Oncogene', 'source': 'Manual', 'evidence': 'Strong'},
        'CCND1': {'role': 'Oncogene', 'source': 'Manual', 'evidence': 'Strong'},
        'EGFR': {'role': 'Oncogene', 'source': 'Manual', 'evidence': 'Strong'},
        'PIK3CA': {'role': 'Oncogene', 'source': 'Manual', 'evidence': 'Strong'},
        'PTEN': {'role': 'Tumor Suppressor', 'source': 'Manual', 'evidence': 'Strong'},
        'TP53': {'role': 'Tumor Suppressor', 'source': 'Manual', 'evidence': 'Strong'},
        'RB1': {'role': 'Tumor Suppressor', 'source': 'Manual', 'evidence': 'Strong'},
        'BRCA1': {'role': 'Tumor Suppressor', 'source': 'Manual', 'evidence': 'Strong'},
        'BRCA2': {'role': 'Tumor Suppressor', 'source': 'Manual', 'evidence': 'Strong'}
    }

    driver_genes_df = pd.DataFrame.from_dict(known_cancer_genes, orient='index').reset_index()
    driver_genes_df.columns = ['gene', 'role', 'source', 'evidence']
    
    os.makedirs("results/tables/drivers", exist_ok=True)
    driver_genes_df.to_csv("results/tables/drivers/comprehensive_driver_genes.tsv", sep='\t', index=False)
    
    return known_cancer_genes, driver_genes_df

def analyze_genome_wide_cnv(cnv_data, sample_info):
    os.makedirs("results/figures/genome_wide", exist_ok=True)
    os.makedirs("results/tables/genome_wide", exist_ok=True)
    
    total_samples = cnv_data.shape[1]
    amp_freq = (cnv_data >= 1).sum(axis=1) / total_samples
    high_amp_freq = (cnv_data == 2).sum(axis=1) / total_samples
    del_freq = (cnv_data <= -1).sum(axis=1) / total_samples
    homo_del_freq = (cnv_data == -2).sum(axis=1) / total_samples
    
    freq_df = pd.DataFrame({
        'gene': cnv_data.index,
        'amp_freq': amp_freq.values,
        'high_amp_freq': high_amp_freq.values,
        'del_freq': del_freq.values,
        'homo_del_freq': homo_del_freq.values
    })
    
    freq_df.to_csv("results/tables/genome_wide/cnv_frequencies.tsv", sep='\t', index=False)

    plt.figure(figsize=(12, 6))
    plt.plot(range(len(freq_df)), freq_df['amp_freq'], 'r-', alpha=0.7, linewidth=1.5, label='Amplification')
    plt.plot(range(len(freq_df)), -freq_df['del_freq'], 'b-', alpha=0.7, linewidth=1.5, label='Deletion')
    plt.xlabel('Genes (ordered by genomic position)')
    plt.ylabel('Alteration Frequency')
    plt.title('Genome-wide CNV Frequency')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig("results/figures/genome_wide/genome_wide_cnv_frequency.png", dpi=300, bbox_inches='tight')
    plt.close()

    subtypes = sample_info['subtype'].unique()
    config = load_config()
    min_samples_per_subtype = config.get("min_samples_per_subtype", 5)

    subtype_summary = []
    
    for subtype in subtypes:
        subtype_samples = sample_info[sample_info['subtype'] == subtype]['sample'].tolist()
        
        if len(subtype_samples) < min_samples_per_subtype:
            continue
            
        subtype_cnv = cnv_data[subtype_samples]

        sub_amp_freq = (subtype_cnv >= 1).sum(axis=1) / len(subtype_samples)
        sub_high_amp_freq = (subtype_cnv == 2).sum(axis=1) / len(subtype_samples)
        sub_del_freq = (subtype_cnv <= -1).sum(axis=1) / len(subtype_samples)
        sub_homo_del_freq = (subtype_cnv == -2).sum(axis=1) / len(subtype_samples)

        for gene in cnv_data.index:
            subtype_summary.append({
                'gene': gene,
                'subtype': subtype,
                'amp_freq': sub_amp_freq[gene],
                'high_amp_freq': sub_high_amp_freq[gene],
                'del_freq': sub_del_freq[gene],
                'homo_del_freq': sub_homo_del_freq[gene]
            })

    subtype_summary_df = pd.DataFrame(subtype_summary)
    subtype_summary_df.to_csv("results/tables/genome_wide/subtype_cnv_frequencies.tsv", sep='\t', index=False)
    
    return freq_df, subtype_summary_df

def identify_subtype_specific_scnas(cnv_data, sample_info):
    os.makedirs("results/figures/subtype_scnas", exist_ok=True)
    os.makedirs("results/tables/subtype_scnas", exist_ok=True)

    config = load_config()
    min_samples_per_subtype = config.get("min_samples_per_subtype", 5)
    fdr_threshold = config.get("fdr_threshold", 0.05)

    subtypes = sample_info['subtype'].unique()
    fisher_results = []

    for subtype in subtypes:
        subtype_samples = sample_info[sample_info['subtype'] == subtype]['sample'].tolist()
        if len(subtype_samples) < min_samples_per_subtype:
            continue
            
        other_samples = sample_info[sample_info['subtype'] != subtype]['sample'].tolist()
        subtype_cnv = cnv_data[subtype_samples]
        other_cnv = cnv_data[other_samples]

        for gene in cnv_data.index:
            if gene not in subtype_cnv.index or gene not in other_cnv.index:
                continue

            amp_in_subtype = (subtype_cnv.loc[gene] >= 1).sum()
            no_amp_in_subtype = len(subtype_samples) - amp_in_subtype
            amp_in_other = (other_cnv.loc[gene] >= 1).sum()
            no_amp_in_other = len(other_samples) - amp_in_other

            contingency_table = np.array([[amp_in_subtype, no_amp_in_subtype],
                                         [amp_in_other, no_amp_in_other]])

            _, p_value = fisher_exact(contingency_table)
            odds_ratio = (amp_in_subtype / no_amp_in_subtype) / (amp_in_other / no_amp_in_other) if no_amp_in_subtype > 0 and amp_in_other > 0 else np.nan

            fisher_results.append({
                'gene': gene,
                'subtype': subtype,
                'alteration': 'amplification',
                'p_value': p_value,
                'odds_ratio': odds_ratio,
                'count_in_subtype': amp_in_subtype,
                'total_in_subtype': len(subtype_samples),
                'frequency_in_subtype': amp_in_subtype / len(subtype_samples) if len(subtype_samples) > 0 else 0,
                'count_in_other': amp_in_other,
                'total_in_other': len(other_samples),
                'frequency_in_other': amp_in_other / len(other_samples) if len(other_samples) > 0 else 0
            })

            del_in_subtype = (subtype_cnv.loc[gene] <= -1).sum()
            no_del_in_subtype = len(subtype_samples) - del_in_subtype
            del_in_other = (other_cnv.loc[gene] <= -1).sum()
            no_del_in_other = len(other_samples) - del_in_other

            contingency_table = np.array([[del_in_subtype, no_del_in_subtype],
                                         [del_in_other, no_del_in_other]])

            _, p_value = fisher_exact(contingency_table)
            odds_ratio = (del_in_subtype / no_del_in_subtype) / (del_in_other / no_del_in_other) if no_del_in_subtype > 0 and del_in_other > 0 else np.nan

            fisher_results.append({
                'gene': gene,
                'subtype': subtype,
                'alteration': 'deletion',
                'p_value': p_value,
                'odds_ratio': odds_ratio,
                'count_in_subtype': del_in_subtype,
                'total_in_subtype': len(subtype_samples),
                'frequency_in_subtype': del_in_subtype / len(subtype_samples) if len(subtype_samples) > 0 else 0,
                'count_in_other': del_in_other,
                'total_in_other': len(other_samples),
                'frequency_in_other': del_in_other / len(other_samples) if len(other_samples) > 0 else 0
            })

    fisher_df = pd.DataFrame(fisher_results)
    correction_results = multipletests(fisher_df['p_value'], method='fdr_bh')
    fisher_df['fdr'] = correction_results[1]

    fisher_df.to_csv("results/tables/subtype_scnas/fisher_test_results.tsv", sep='\t', index=False)

    sig_results = fisher_df[fisher_df['fdr'] < fdr_threshold].sort_values('fdr')
    sig_results.to_csv("results/tables/subtype_scnas/significant_scnas.tsv", sep='\t', index=False)
    
    return fisher_df, sig_results

def analyze_cnv_expression_correlation(cnv_data, expr_data, sample_info):
    os.makedirs("results/figures/correlation", exist_ok=True)
    os.makedirs("results/tables/correlation", exist_ok=True)
    
    common_samples = list(set(cnv_data.columns) & set(expr_data.columns))
    common_genes = list(set(cnv_data.index) & set(expr_data.index))
    
    correlation_results = {'all': {}}
    
    if len(common_genes) > 1000:
        key_cancer_genes = ['ERBB2', 'MYC', 'CCND1', 'EGFR', 'PIK3CA', 'PTEN', 'TP53', 'RB1']
        selected_genes = [g for g in key_cancer_genes if g in common_genes]
        remaining_genes = [g for g in common_genes if g not in selected_genes]
        remaining_needed = 1000 - len(selected_genes)
        random_genes = np.random.choice(remaining_genes, min(remaining_needed, len(remaining_genes)), replace=False)
        selected_genes.extend(random_genes)
    else:
        selected_genes = common_genes
    
    all_correlations = []
    
    for gene in selected_genes:
        cnv_values = cnv_data.loc[gene, common_samples]
        expr_values = expr_data.loc[gene, common_samples]
        
        corr, p_value = safe_pearsonr(cnv_values, expr_values)
        
        correlation_results['all'][gene] = corr
        all_correlations.append(corr)
        
        if gene in ['ERBB2', 'MYC', 'CCND1', 'PTEN', 'TP53', 'RB1', 'EGFR', 'PIK3CA']:
            plt.figure(figsize=(8, 6))
            plt.hexbin(cnv_values, expr_values, gridsize=20, cmap='viridis', alpha=0.8)
            plt.colorbar(label='Sample Count')
            plt.xlabel('Copy Number Value')
            plt.ylabel('Expression Level (log2)')
            plt.title(f'{gene} CNV vs Expression (r = {corr:.3f}, p = {p_value:.1e})')
            
            if len(cnv_values) > 1:
                z = np.polyfit(cnv_values, expr_values, 1)
                p = np.poly1d(z)
                x_range = np.linspace(min(cnv_values), max(cnv_values), 100)
                plt.plot(x_range, p(x_range), 'r--', linewidth=2)
            
            plt.text(0.05, 0.95, f'Pearson r = {corr:.3f}\np-value = {p_value:.2e}\nn = {len(common_samples)}', 
                    transform=plt.gca().transAxes, fontsize=10, 
                    bbox=dict(facecolor='white', alpha=0.7))
            
            plt.tight_layout()
            plt.savefig(f"results/figures/correlation/{gene}_cnv_expr_correlation.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    plt.figure(figsize=(10, 6))
    sns.histplot(all_correlations, bins=30, kde=True)
    plt.xlabel('CNV-Expression Correlation (Pearson r)')
    plt.ylabel('Frequency')
    plt.title('Distribution of CNV-Expression Correlations')
    
    median_corr = np.median(all_correlations)
    mean_corr = np.mean(all_correlations)
    plt.axvline(median_corr, color='r', linestyle='--', alpha=0.7, label=f'Median: {median_corr:.3f}')
    plt.axvline(mean_corr, color='g', linestyle='--', alpha=0.7, label=f'Mean: {mean_corr:.3f}')
    plt.legend()
    
    plt.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig("results/figures/correlation/correlation_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    subtypes = sample_info['subtype'].unique()
    config = load_config()
    min_samples_per_subtype = config.get("min_samples_per_subtype", 5)
    
    for subtype in subtypes:
        subtype_samples = sample_info[sample_info['subtype'] == subtype]['sample'].tolist()
        common_subtype_samples = [s for s in subtype_samples if s in common_samples]
        
        if len(common_subtype_samples) < min_samples_per_subtype:
            continue
        
        correlation_results[subtype] = {}
        
        key_cancer_genes = ['ERBB2', 'MYC', 'CCND1', 'EGFR', 'PIK3CA', 'PTEN', 'TP53', 'RB1']
        subtype_genes = [g for g in key_cancer_genes if g in common_genes]
        
        for gene in subtype_genes:
            cnv_values = cnv_data.loc[gene, common_subtype_samples]
            expr_values = expr_data.loc[gene, common_subtype_samples]
            
            corr, p_value = safe_pearsonr(cnv_values, expr_values)
            correlation_results[subtype][gene] = corr
    
    correlation_summary = []
    
    for gene, corr in correlation_results['all'].items():
        result = {'gene': gene, 'all_samples': corr}
        
        for subtype in subtypes:
            if subtype in correlation_results and gene in correlation_results[subtype]:
                result[subtype] = correlation_results[subtype][gene]
            else:
                result[subtype] = np.nan
        
        correlation_summary.append(result)
    
    correlation_df = pd.DataFrame(correlation_summary)
    correlation_df.to_csv("results/tables/correlation/cnv_expression_correlation.tsv", sep='\t', index=False)
    
    return correlation_results

def identify_driver_genes(cnv_data, expr_data, fisher_results, correlation_results, known_cancer_genes):
    os.makedirs("results/tables/drivers", exist_ok=True)
    os.makedirs("results/figures/drivers", exist_ok=True)
    
    config = load_config()
    ml_confidence_threshold = config.get("ml_confidence_threshold", 0.6)
    known_driver_bonus = config.get("known_driver_bonus", 3)
    expression_correlation_threshold = config.get("expression_correlation_threshold", 0.3)
    
    signif_altered = {}
    
    if fisher_results is not None:
        signif_results = fisher_results[fisher_results['fdr'] < 0.05]
        
        for _, row in signif_results.iterrows():
            gene = row['gene']
            alt = row['alteration']
            subtype = row['subtype']
            
            if gene not in signif_altered:
                signif_altered[gene] = {'amplification': [], 'deletion': []}
            
            signif_altered[gene][alt].append(subtype)
    
    cis_correlated = {}
    
    if correlation_results is not None:
        all_corrs = correlation_results.get('all', {})
        
        for gene, corr in all_corrs.items():
            if abs(corr) > expression_correlation_threshold:
                cis_correlated[gene] = corr
    
    driver_prediction_scores = {}
    
    if len(known_cancer_genes) >= 20:
        features = []
        labels = []
        gene_list = []
        
        for gene in cnv_data.index:
            if gene not in expr_data.index:
                continue
                
            amp_freq = (cnv_data.loc[gene] >= 1).mean()
            high_amp_freq = (cnv_data.loc[gene] == 2).mean()
            del_freq = (cnv_data.loc[gene] <= -1).mean()
            homo_del_freq = (cnv_data.loc[gene] == -2).mean()
            
            mean_expr = expr_data.loc[gene].mean()
            var_expr = expr_data.loc[gene].var()
            
            common_samples = list(set(cnv_data.columns) & set(expr_data.columns))
            
            cnv_values = cnv_data.loc[gene, common_samples]
            expr_values = expr_data.loc[gene, common_samples]
            
            cnv_expr_corr, _ = safe_pearsonr(cnv_values, expr_values)
            
            sig_amp = 1 if gene in signif_altered and signif_altered[gene]['amplification'] else 0
            sig_del = 1 if gene in signif_altered and signif_altered[gene]['deletion'] else 0
            
            feature_vector = [
                amp_freq, high_amp_freq, del_freq, homo_del_freq,
                mean_expr, var_expr, cnv_expr_corr, sig_amp, sig_del
            ]
            
            features.append(feature_vector)
            gene_list.append(gene)
            
            labels.append(1 if gene in known_cancer_genes else 0)
        
        X = np.array(features)
        y = np.array(labels)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        
        clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        clf.fit(X_train, y_train)
        
        pred_probs = clf.predict_proba(X)[:, 1]
        
        for gene, score in zip(gene_list, pred_probs):
            driver_prediction_scores[gene] = score
            
        test_probs = clf.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, test_probs)
        fpr, tpr, _ = roc_curve(y_test, test_probs)
        roc_auc = auc(fpr, tpr)
    
    driver_candidates = []
    
    for gene in cnv_data.index:
        if gene not in expr_data.index:
            continue
        
        evidence_score = 0
        evidence_sources = []
        alteration_type = 'Unknown'
        
        if gene in known_cancer_genes:
            role = known_cancer_genes[gene]['role']
            source = known_cancer_genes[gene]['source']
            evidence = known_cancer_genes[gene]['evidence']
            
            if evidence == 'Strong':
                evidence_score += known_driver_bonus
            elif evidence == 'Moderate':
                evidence_score += 2
            else:
                evidence_score += 1
                
            evidence_sources.append(f"Known cancer gene ({source})")
            alteration_type = role
        
        if gene in signif_altered:
            if signif_altered[gene]['amplification']:
                evidence_score += len(signif_altered[gene]['amplification'])
                evidence_sources.append(f"Significant amplification in {', '.join(signif_altered[gene]['amplification'])}")
                if alteration_type == 'Unknown':
                    alteration_type = 'Oncogene'
            
            if signif_altered[gene]['deletion']:
                evidence_score += len(signif_altered[gene]['deletion'])
                evidence_sources.append(f"Significant deletion in {', '.join(signif_altered[gene]['deletion'])}")
                if alteration_type == 'Unknown':
                    alteration_type = 'Tumor Suppressor'
        
        if gene in cis_correlated:
            corr = cis_correlated[gene]
            if abs(corr) > 0.5:
                evidence_score += 2
            else:
                evidence_score += 1
            
            evidence_sources.append(f"Cis-correlation: {corr:.3f}")
        
        if gene in driver_prediction_scores:
            ml_score = driver_prediction_scores[gene]
            if ml_score > 0.8:
                evidence_score += 3
                evidence_sources.append(f"ML prediction: {ml_score:.3f} (high confidence)")
            elif ml_score > ml_confidence_threshold:
                evidence_score += 2
                evidence_sources.append(f"ML prediction: {ml_score:.3f} (medium confidence)")
            elif ml_score > 0.4:
                evidence_score += 1
                evidence_sources.append(f"ML prediction: {ml_score:.3f} (low confidence)")
        
        if evidence_score > 0:
            driver_candidates.append({
                'gene': gene,
                'evidence_score': evidence_score,
                'evidence_sources': '; '.join(evidence_sources),
                'alteration_type': alteration_type,
                'ml_score': driver_prediction_scores.get(gene, None)
            })
    
    driver_df = pd.DataFrame(driver_candidates).sort_values('evidence_score', ascending=False)
    driver_df.to_csv("results/tables/drivers/driver_candidates.tsv", sep='\t', index=False)
    
    plt.figure(figsize=(12, 10))
    top_drivers = driver_df.head(20)
    
    color_map = {
        'Oncogene': '#d73027',
        'Tumor Suppressor': '#4575b4',
        'Lineage Factor': '#4daf4a',
        'Unknown': '#999999'
    }
    
    bar_colors = [color_map.get(alt, '#999999') for alt in top_drivers['alteration_type']]
    bars = plt.barh(top_drivers['gene'], top_drivers['evidence_score'], color=bar_colors, alpha=0.8)
    
    for i, bar in enumerate(bars):
        plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                f"{int(top_drivers.iloc[i]['evidence_score'])}", 
                va='center', fontsize=10)
    
    plt.xlabel('Evidence Score')
    plt.ylabel('Gene')
    plt.title('Top 20 Candidate Driver Genes')
    plt.xlim(0, max(top_drivers['evidence_score']) * 1.15)
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    
    legend_elements = [
        Patch(facecolor='#d73027', label='Oncogene'),
        Patch(facecolor='#4575b4', label='Tumor Suppressor'),
        Patch(facecolor='#4daf4a', label='Lineage Factor'),
        Patch(facecolor='#999999', label='Unknown')
    ]
    plt.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig("results/figures/drivers/top_driver_candidates.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return driver_df

def perform_pathway_analysis(driver_df, trans_correlation_results):
    os.makedirs("results/tables/pathways", exist_ok=True)
    os.makedirs("results/figures/pathways", exist_ok=True)
    
    pathways = {
        'PI3K-AKT-mTOR': ['PIK3CA', 'PIK3CB', 'PIK3CD', 'PIK3R1', 'AKT1', 'AKT2', 'AKT3', 'MTOR', 'PTEN', 'TSC1', 'TSC2', 'RICTOR', 'RPTOR'],
        'RAS-MAPK': ['KRAS', 'HRAS', 'NRAS', 'BRAF', 'RAF1', 'MAP2K1', 'MAP2K2', 'MAPK1', 'MAPK3', 'MAPK8', 'MAPK9', 'MAPK10'],
        'Cell Cycle': ['CCND1', 'CCNE1', 'CCNA2', 'CCNB1', 'CDK1', 'CDK2', 'CDK4', 'CDK6', 'RB1', 'E2F1', 'E2F2', 'E2F3'],
        'DNA Repair': ['BRCA1', 'BRCA2', 'PALB2', 'RAD51', 'ATM', 'ATR', 'TP53', 'CHEK1', 'CHEK2', 'MDM2'],
        'Apoptosis': ['BCL2', 'BCL2L1', 'BAX', 'BAK1', 'CASP3', 'CASP8', 'CASP9', 'BID', 'PMAIP1', 'MCL1']
    }
    
    if driver_df is not None and not driver_df.empty:
        top_drivers = list(driver_df.head(30)['gene'])
        
        pathway_results = []
        
        config = load_config()
        evidence_score_threshold = config.get("evidence_score_threshold", 3)
        
        for pathway, genes in pathways.items():
            overlap = set(top_drivers).intersection(set(genes))
            
            if len(overlap) > 0:
                if len(overlap) > 1:
                    all_genes_count = len(set(driver_df['gene']))
                    pathway_size = len(genes)
                    driver_count = len(top_drivers)
                    overlap_count = len(overlap)
                    
                    p_value = hypergeom.sf(overlap_count-1, all_genes_count, pathway_size, driver_count)
                else:
                    p_value = 1.0
                
                pathway_results.append({
                    'pathway': pathway,
                    'overlap_count': len(overlap),
                    'overlap_genes': ', '.join(overlap),
                    'overlap_ratio': len(overlap) / len(genes),
                    'driver_ratio': len(overlap) / len(top_drivers),
                    'p_value': p_value
                })
        
        if pathway_results:
            pathway_df = pd.DataFrame(pathway_results).sort_values('overlap_count', ascending=False)
            
            if len(pathway_df) > 1:
                correction_results = multipletests(pathway_df['p_value'], method='fdr_bh')
                pathway_df['fdr'] = correction_results[1]
            else:
                pathway_df['fdr'] = pathway_df['p_value']
            
            pathway_df.to_csv("results/tables/pathways/driver_pathway_enrichment.tsv", sep='\t', index=False)
            
            plt.figure(figsize=(12, 8))
            
            bars = plt.barh(pathway_df['pathway'], pathway_df['overlap_count'], color='#6baed6', alpha=0.8)
            
            for i, bar in enumerate(bars):
                plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                        f"{pathway_df.iloc[i]['overlap_count']} genes", 
                        va='center', fontsize=10)
                
                if pathway_df.iloc[i]['overlap_count'] <= 3:
                    genes_text = pathway_df.iloc[i]['overlap_genes']
                    plt.text(0.5, bar.get_y() + bar.get_height()/2, 
                            f"({genes_text})", 
                            va='center', fontsize=9, color='#333333')
            
            plt.xlabel('Number of Driver Genes')
            plt.ylabel('Pathway')
            plt.title('Driver Gene Enrichment in Canonical Pathways')
            plt.grid(axis='x', alpha=0.3, linestyle='--')
            plt.tight_layout()
            plt.savefig("results/figures/pathways/driver_pathway_enrichment.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            return pathway_df
    
    return pd.DataFrame()

def create_integrated_visualizations(cnv_data, expr_data, sample_info, driver_df, protein_data=None):
    os.makedirs("results/figures/integrated", exist_ok=True)
    
    if driver_df is not None and not driver_df.empty:
        top_drivers = list(driver_df.head(10)['gene'])
    else:
        top_drivers = ['ERBB2', 'MYC', 'CCND1', 'EGFR', 'PIK3CA', 'PTEN', 'TP53', 'RB1']
    
    common_genes = list(set(cnv_data.index) & set(expr_data.index))
    common_drivers = [gene for gene in top_drivers if gene in common_genes]
    
    common_samples = list(set(cnv_data.columns) & set(expr_data.columns))
    
    sample_subtypes = {}
    for _, row in sample_info.iterrows():
        if 'sample' in row and 'subtype' in row:
            sample_subtypes[row['sample']] = row['subtype']
    
    subtyped_samples = [s for s in common_samples if s in sample_subtypes]
    
    driver_cnv = cnv_data.loc[common_drivers, subtyped_samples]
    driver_expr = expr_data.loc[common_drivers, subtyped_samples]
    
    driver_protein = None
    if protein_data is not None:
        protein_matches = {}
        for gene in common_drivers:
            matches = [p for p in protein_data.index if gene.lower() in p.lower()]
            if matches:
                protein_matches[gene] = matches[0]
        
        if protein_matches:
            protein_rows = []
            
            for gene, protein in protein_matches.items():
                if protein in protein_data.index:
                    protein_rows.append(protein_data.loc[protein, subtyped_samples])
            
            if protein_rows:
                driver_protein = pd.DataFrame(protein_rows, index=[p for p in protein_matches.values()])
    
    for driver in common_drivers[:5]:
        if driver not in expr_data.index:
            continue
            
        plt.figure(figsize=(10, 8))
        
        driver_cnv_values = cnv_data.loc[driver, subtyped_samples]
        driver_expr_values = expr_data.loc[driver, subtyped_samples]
        
        for subtype, color in SUBTYPE_COLORS.items():
            subtype_samples = [s for s in subtyped_samples if sample_subtypes.get(s, '') == subtype]
            
            if not subtype_samples:
                continue
            
            plt.scatter(
                driver_cnv_values[subtype_samples],
                driver_expr_values[subtype_samples],
                color=color,
                label=f"{subtype} (n={len(subtype_samples)})",
                alpha=0.7,
                s=50,
                edgecolors='black',
                linewidths=0.5
            )
        
        plt.xlabel('Copy Number Value')
        plt.ylabel('Expression (log2)')
        plt.title(f'{driver} CNV vs Expression by Subtype')
        
        plt.legend(fontsize=10)
        
        corr, p_value = safe_pearsonr(driver_cnv_values, driver_expr_values)
        
        z = np.polyfit(driver_cnv_values, driver_expr_values, 1)
        p = np.poly1d(z)
        cnv_range = np.linspace(min(driver_cnv_values), max(driver_cnv_values), 100)
        plt.plot(cnv_range, p(cnv_range), "k--", alpha=0.8, linewidth=2)
        
        plt.text(0.05, 0.95, f'Correlation: {corr:.3f} (p={p_value:.1e})', 
                transform=plt.gca().transAxes, fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f"results/figures/integrated/{driver}_subtype_cnv_expr.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    return {
        'driver_cnv': driver_cnv,
        'driver_expr': driver_expr,
        'driver_protein': driver_protein
    }

def analyze_survival(merged_data, output_dir="results/figures/survival"):
    os.makedirs(output_dir, exist_ok=True)
    
    if 'OS' in merged_data.columns and 'OS.time' in merged_data.columns:
        if 'silhouette' in merged_data.columns:
            has_silhouette = True
            
            silhouette_threshold = 0.25
            merged_data['confidence'] = merged_data['silhouette'].apply(
                lambda x: 'High' if x >= silhouette_threshold else 'Low'
            )
        else:
            has_silhouette = False
        
        kmf = KaplanMeierFitter()
        
        plt.figure(figsize=(12, 8))
        
        for subtype in merged_data['subtype'].unique():
            subset = merged_data[merged_data['subtype'] == subtype]
            
            if len(subset) < 5:
                continue
            
            kmf.fit(
                durations=subset['OS.time'],
                event_observed=subset['OS'].astype(int),
                label=f"{subtype} (n={len(subset)})"
            )
            kmf.plot(ci_show=False)
        
        plt.title('Overall Survival by Molecular Subtype')
        plt.xlabel('Time (days)')
        plt.ylabel('Survival Probability')
        plt.grid(alpha=0.3)
        plt.legend()
        plt.savefig(f"{output_dir}/overall_survival.png")
        plt.close()
        
        if has_silhouette:
            for subtype in merged_data['subtype'].unique():
                high_conf = merged_data[(merged_data['subtype'] == subtype) & 
                                      (merged_data['confidence'] == 'High')]
                low_conf = merged_data[(merged_data['subtype'] == subtype) & 
                                     (merged_data['confidence'] == 'Low')]
                
                if len(high_conf) < 5 or len(low_conf) < 5:
                    continue
                
                plt.figure(figsize=(10, 6))
                
                kmf.fit(
                    durations=high_conf['OS.time'],
                    event_observed=high_conf['OS'].astype(int),
                    label=f"High confidence (n={len(high_conf)})"
                )
                kmf.plot(ci_show=False, color='darkblue')
                
                kmf.fit(
                    durations=low_conf['OS.time'],
                    event_observed=low_conf['OS'].astype(int),
                    label=f"Low confidence (n={len(low_conf)})"
                )
                kmf.plot(ci_show=False, color='lightblue', linestyle='--')
                
                results = logrank_test(
                    durations_A=high_conf['OS.time'],
                    durations_B=low_conf['OS.time'],
                    event_observed_A=high_conf['OS'].astype(int),
                    event_observed_B=low_conf['OS'].astype(int)
                )
                
                plt.text(0.05, 0.15, f'Logrank p-value: {results.p_value:.4f}', 
                       transform=plt.gca().transAxes, fontsize=12)
                
                plt.title(f'Overall Survival for {subtype} by Classification Confidence')
                plt.xlabel('Time (days)')
                plt.ylabel('Survival Probability')
                plt.grid(alpha=0.3)
                plt.legend()
                plt.tight_layout()
                plt.savefig(f"{output_dir}/{subtype.replace('/', '_')}_survival_by_confidence.png")
                plt.close()
        
        subtypes = merged_data['subtype'].unique()
        results = []
        
        for i, s1 in enumerate(subtypes):
            subset1 = merged_data[merged_data['subtype'] == s1]
            
            if len(subset1) < 5:
                continue
                
            for j, s2 in enumerate(subtypes[i+1:], i+1):
                subset2 = merged_data[merged_data['subtype'] == s2]
                
                if len(subset2) < 5:
                    continue
                
                test = logrank_test(
                    durations_A=subset1['OS.time'],
                    durations_B=subset2['OS.time'],
                    event_observed_A=subset1['OS'].astype(int),
                    event_observed_B=subset2['OS'].astype(int)
                )
                
                results.append({
                    'subtype1': s1,
                    'subtype2': s2,
                    'p_value': test.p_value,
                    'n1': len(subset1),
                    'n2': len(subset2)
                })
        
        if results:
            results_df = pd.DataFrame(results)
            results_df.to_csv(f"{output_dir}/logrank_results.tsv", sep='\t', index=False)
        
        return True
    else:
        return False

def calculate_cnv_risk_score(cnv_data, expr_data, sample_info, driver_df, survival_data=None):
    os.makedirs("results/figures/risk_score", exist_ok=True)
    os.makedirs("results/tables/risk_score", exist_ok=True)
    
    sample_samples = set(sample_info['sample'].values)
    common_samples = list(set(cnv_data.columns) & set(expr_data.columns) & sample_samples)
    
    if driver_df is not None and not driver_df.empty:
        top_drivers = list(driver_df.head(20)['gene'])
        
        driver_roles = {}
        for _, row in driver_df.head(20).iterrows():
            gene = row['gene']
            
            if 'alteration_type' in row:
                role = row['alteration_type']
                driver_roles[gene] = role
            else:
                if 'evidence_sources' in row:
                    evidence = row['evidence_sources']
                    if 'Oncogene' in evidence:
                        driver_roles[gene] = 'Oncogene'
                    elif 'Tumor Suppressor' in evidence:
                        driver_roles[gene] = 'Tumor Suppressor'
                    else:
                        if gene in expr_data.index and gene in cnv_data.index:
                            corr, _ = safe_pearsonr(cnv_data.loc[gene, common_samples], expr_data.loc[gene, common_samples])
                            driver_roles[gene] = 'Oncogene' if corr > 0 else 'Tumor Suppressor'
                        else:
                            driver_roles[gene] = 'Unknown'
    else:
        top_drivers = ['ERBB2', 'MYC', 'CCND1', 'EGFR', 'PIK3CA', 'PTEN', 'TP53', 'RB1']
        driver_roles = {
            'ERBB2': 'Oncogene', 'MYC': 'Oncogene', 'CCND1': 'Oncogene', 'EGFR': 'Oncogene', 
            'PIK3CA': 'Oncogene', 'PTEN': 'Tumor Suppressor', 'TP53': 'Tumor Suppressor', 'RB1': 'Tumor Suppressor'
        }
    
    top_drivers = [gene for gene in top_drivers if gene in cnv_data.index]
    
    if not top_drivers:
        return None
    
    risk_scores = pd.Series(0, index=common_samples)
    
    for gene in top_drivers:
        if gene not in driver_roles:
            continue
            
        gene_cnv = cnv_data.loc[gene, common_samples]
        
        if driver_roles[gene] == 'Oncogene':
            risk_scores += gene_cnv
        elif driver_roles[gene] == 'Tumor Suppressor':
            risk_scores -= gene_cnv
        else:
            risk_scores += abs(gene_cnv)
    
    risk_scores = (risk_scores - risk_scores.mean()) / risk_scores.std()
    
    risk_df = pd.DataFrame({
        'sample': risk_scores.index,
        'risk_score': risk_scores.values
    })
    
    risk_df = risk_df.merge(
        sample_info[['sample', 'subtype']], 
        on='sample', 
        how='left'
    )
    
    risk_df['risk_group'] = pd.qcut(
        risk_df['risk_score'],
        q=3,
        labels=['Low', 'Intermediate', 'High']
    )
    
    risk_df.to_csv("results/tables/risk_score/cnv_risk_scores.tsv", sep='\t', index=False)
    
    plt.figure(figsize=(12, 6))
    
    for subtype in risk_df['subtype'].unique():
        if pd.isna(subtype):
            continue
            
        subtype_data = risk_df[risk_df['subtype'] == subtype]
        
        if len(subtype_data) < 5:
            continue
            
        sns.kdeplot(
            subtype_data['risk_score'],
            label=f"{subtype} (n={len(subtype_data)})",
            color=SUBTYPE_COLORS.get(subtype, None),
            alpha=0.7
        )
    
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    plt.grid(alpha=0.3, linestyle='--')
    plt.xlabel('CNV Risk Score')
    plt.ylabel('Density')
    plt.title('Distribution of CNV Risk Scores by Subtype')
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/figures/risk_score/risk_score_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    if survival_data is not None and 'OS' in survival_data.columns and 'OS.time' in survival_data.columns:
        survival_risk = risk_df.merge(
            survival_data[['sample', 'OS', 'OS.time']],
            on='sample',
            how='inner'
        )
        
        if len(survival_risk) > 0:
            plt.figure(figsize=(10, 6))
            
            kmf = KaplanMeierFitter()
            
            for risk_group in ['Low', 'Intermediate', 'High']:
                group_data = survival_risk[survival_risk['risk_group'] == risk_group]
                
                if len(group_data) < 5:
                    continue
                    
                kmf.fit(
                    durations=group_data['OS.time'],
                    event_observed=group_data['OS'].astype(int),
                    label=f"{risk_group} Risk (n={len(group_data)})"
                )
                
                color = 'green' if risk_group == 'Low' else 'orange' if risk_group == 'Intermediate' else 'red'
                kmf.plot(ci_show=False, color=color)
            
            high_risk = survival_risk[survival_risk['risk_group'] == 'High']
            low_risk = survival_risk[survival_risk['risk_group'] == 'Low']
            
            if len(high_risk) >= 5 and len(low_risk) >= 5:
                results = logrank_test(
                    durations_A=high_risk['OS.time'],
                    durations_B=low_risk['OS.time'],
                    event_observed_A=high_risk['OS'].astype(int),
                    event_observed_B=low_risk['OS'].astype(int)
                )
                
                plt.text(0.05, 0.15, f'Log-rank p-value (High vs Low): {results.p_value:.4f}', 
                       transform=plt.gca().transAxes, fontsize=12,
                       bbox=dict(facecolor='white', alpha=0.7))
            
            plt.title('Overall Survival by CNV Risk Group')
            plt.xlabel('Time (days)')
            plt.ylabel('Survival Probability')
            plt.grid(alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig("results/figures/risk_score/risk_group_survival.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    return risk_df

def run_breast_cancer_cnv_analysis():
    start_time = time.time()
    
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)
    
    config = load_config()
    data_dict = load_processed_data()
    
    if data_dict.get('gene_locations') is None:
        data_dict['gene_locations'] = map_genes_to_chromosomes(data_dict['cnv_data'])
    
    if config.get("impute_missing_values", True):
        data_dict = impute_missing_data(data_dict)
    
    cnv_data = data_dict['cnv_data']
    expr_data = data_dict['expr_data']
    sample_info = data_dict['sample_info']
    protein_data = data_dict.get('protein_data')
    gene_locations = data_dict.get('gene_locations')
    external_data = data_dict.get('external_data')
    
    known_cancer_genes, driver_genes_df = load_driver_gene_databases()
    freq_df, subtype_freq_df = analyze_genome_wide_cnv(cnv_data, sample_info)
    fisher_results, sig_results = identify_subtype_specific_scnas(cnv_data, sample_info)
    correlation_results = analyze_cnv_expression_correlation(cnv_data, expr_data, sample_info)
    
    driver_df = identify_driver_genes(cnv_data, expr_data, fisher_results, correlation_results, known_cancer_genes)
    pathway_results = perform_pathway_analysis(driver_df, None)
    integrated_results = create_integrated_visualizations(cnv_data, expr_data, sample_info, driver_df, protein_data)
    risk_df = calculate_cnv_risk_score(cnv_data, expr_data, sample_info, driver_df)
    survival_success = analyze_survival(sample_info)
    
    execution_time = time.time() - start_time
    
    summary = {
        'execution_time': execution_time,
        'sample_count': cnv_data.shape[1],
        'gene_count': cnv_data.shape[0],
        'subtypes': list(sample_info['subtype'].unique()),
        'top_drivers': list(driver_df.head(10)['gene']) if driver_df is not None else [],
        'significant_scnas': len(sig_results) if sig_results is not None else 0
    }
    
    return {
        'success': True,
        'summary': summary
    }

if __name__ == "__main__":
    run_breast_cancer_cnv_analysis()
