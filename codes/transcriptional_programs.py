import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import yaml
from scipy.stats import ttest_ind, pearsonr
from scipy.cluster.hierarchy import linkage
from statsmodels.stats.multitest import multipletests
from sklearn.preprocessing import StandardScaler
import networkx as nx
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
import statsmodels.api as sm

def load_config():
    if os.path.exists("config.yaml"):
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
    else:
        config = {
            "use_external_validation": True,
            "impute_missing_values": True,
            "min_samples_per_group": 5,
            "min_genes_per_program": 10,
            "p_value_threshold": 0.05,
            "fold_change_threshold": 1.0,
            "max_genes_to_analyze": 5000,
            "correlation_threshold": 0.3,
            "max_network_genes": 100,
            "network_layout": "spring",
            "use_multivariate_models": True,
            "use_gsea": True,
            "use_network_inference": True,
            "use_clinical_integration": True,
            "use_causal_inference": False,
            "use_deep_learning": True,
            "batch_processing": True,
            "batch_size": 1000,
            "validation_min_genes": 3,
            "create_interactive_plots": False,
            "save_svg_format": True,
            "figure_dpi": 300
        }
        with open("config.yaml", "w") as f:
            yaml.dump(config, f, default_flow_style=False)
    return config

def load_processed_data():
    data_paths = {
        "cnv_data": "data/processed/matched_cnv.tsv",
        "expr_data": "data/processed/expression_mapped.tsv",
        "sample_info": "data/processed/clinical_with_subtypes.tsv",
        "driver_df": "results/tables/drivers/driver_candidates.tsv"
    }
    
    optional_paths = {
        "correlation_df": "results/tables/correlation/cnv_expression_correlation_summary.tsv",
        "gene_locations": "data/processed/gene_locations.tsv",
        "protein_data": "data/processed/matched_rppa.tsv",
        "survival_data": "data/processed/survival.tsv"
    }
    
    data_dict = {}
    
    for key, path in data_paths.items():
        if key == "sample_info":
            data_dict[key] = pd.read_csv(path, sep='\t')
        else:
            data_dict[key] = pd.read_csv(path, sep='\t', index_col=0)
    
    for key, path in optional_paths.items():
        if os.path.exists(path):
            if key == "survival_data":
                data_dict[key] = pd.read_csv(path, sep='\t')
            else:
                data_dict[key] = pd.read_csv(path, sep='\t', index_col=0)
    
    sample_subtypes = {}
    for _, row in data_dict['sample_info'].iterrows():
        if 'sample' in row and 'subtype' in row:
            sample_subtypes[row['sample']] = row['subtype']
    
    data_dict['sample_subtypes'] = sample_subtypes
    
    return data_dict

def load_external_validation_data():
    metabric_files = {
        "cnv": "data/external/metabric_cnv.tsv",
        "expression": "data/external/metabric_expression.tsv",
        "clinical": "data/external/metabric_clinical.tsv"
    }
    
    external_data = {}
    for data_type, file_path in metabric_files.items():
        if os.path.exists(file_path):
            if data_type == "clinical":
                data = pd.read_csv(file_path, sep='\t')
            else:
                data = pd.read_csv(file_path, sep='\t', index_col=0)
            external_data[data_type] = data
    
    if "clinical" in external_data and "subtype" in external_data["clinical"].columns:
        external_subtypes = {}
        for _, row in external_data["clinical"].iterrows():
            if 'sample' in row and 'subtype' in row:
                external_subtypes[row['sample']] = row['subtype']
        external_data['sample_subtypes'] = external_subtypes
    
    return external_data if external_data else None

def safe_pearsonr(x, y):
    if len(x) < 3 or len(y) < 3:
        return np.nan, np.nan
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan
    return pearsonr(x, y)

def identify_transcriptional_programs(data_dict, config):
    os.makedirs("results/figures/transcriptional_programs", exist_ok=True)
    os.makedirs("results/tables/transcriptional_programs", exist_ok=True)
    
    cnv_data = data_dict['cnv_data']
    expr_data = data_dict['expr_data']
    driver_df = data_dict['driver_df']
    sample_subtypes = data_dict['sample_subtypes']
    
    min_samples = config.get("min_samples_per_group", 5)
    p_threshold = config.get("p_value_threshold", 0.05)
    fc_threshold = config.get("fold_change_threshold", 1.0)
    max_genes = config.get("max_genes_to_analyze", 5000)
    use_multivariate = config.get("use_multivariate_models", False)
    batch_processing = config.get("batch_processing", True)
    batch_size = config.get("batch_size", 1000)
    
    if 'gene' in driver_df.columns:
        top_drivers = list(driver_df.sort_values('evidence_score', ascending=False).head(20)['gene'])
    else:
        top_drivers = list(driver_df.sort_values('evidence_score', ascending=False).head(20).index)
    
    common_drivers = [gene for gene in top_drivers if gene in cnv_data.index and gene in expr_data.index]
    
    if not common_drivers:
        default_drivers = ['ERBB2', 'MYC', 'CCND1', 'EGFR', 'PIK3CA', 'PTEN', 'TP53', 'RB1']
        common_drivers = [gene for gene in default_drivers if gene in cnv_data.index and gene in expr_data.index]
    
    common_samples = list(set(cnv_data.columns) & set(expr_data.columns))
    common_samples = [s for s in common_samples if s in sample_subtypes]
    
    cnv_filtered = cnv_data[common_samples]
    expr_filtered = expr_data[common_samples]
    
    program_results = {}
    
    for driver in common_drivers:
        driver_cnv = cnv_filtered.loc[driver]
        driver_expr = expr_filtered.loc[driver]
        
        amplified_samples = [s for s in common_samples if driver_cnv[s] >= 1]
        neutral_samples = [s for s in common_samples if -1 < driver_cnv[s] < 1]
        deleted_samples = [s for s in common_samples if driver_cnv[s] <= -1]
        
        if len(amplified_samples) < min_samples and len(deleted_samples) < min_samples:
            continue
        
        if len(amplified_samples) >= min_samples:
            group1 = amplified_samples
            group1_name = 'Amplified'
            group2 = neutral_samples
            group2_name = 'Neutral'
            alteration_type = 'Amplification'
        elif len(deleted_samples) >= min_samples:
            group1 = deleted_samples
            group1_name = 'Deleted'
            group2 = neutral_samples
            group2_name = 'Neutral'
            alteration_type = 'Deletion'
        else:
            continue
                
        if len(group1) < min_samples or len(group2) < min_samples:
            continue
        
        diff_expr_results = []
        
        if use_multivariate:
            samples = group1 + group2
            cnv_status = [1 if s in group1 else 0 for s in samples]
            gene_subset = list(expr_filtered.index)
            if len(gene_subset) > max_genes:
                gene_subset = np.random.choice(gene_subset, max_genes, replace=False)
            
            subtypes = [sample_subtypes.get(s, 'Unknown') for s in samples]
            subtype_dummies = pd.get_dummies(subtypes, drop_first=True)
            
            if batch_processing:
                for i in range(0, len(gene_subset), batch_size):
                    batch_genes = gene_subset[i:i+batch_size]
                    
                    for gene in batch_genes:
                        if gene == driver:
                            continue
                            
                        gene_expr = expr_filtered.loc[gene, samples].values
                        
                        if np.isnan(gene_expr).any() or not np.all(np.isfinite(gene_expr)):
                            continue
                        
                        model_df = pd.DataFrame({
                            'expression': gene_expr,
                            'cnv_status': cnv_status
                        })
                        
                        for col in subtype_dummies.columns:
                            model_df[col] = subtype_dummies[col].values
                        
                        X = sm.add_constant(model_df.drop('expression', axis=1))
                        model = sm.OLS(model_df['expression'], X).fit()
                        
                        cnv_effect = model.params['cnv_status']
                        p_value = model.pvalues['cnv_status']
                        
                        mean1 = np.mean(expr_filtered.loc[gene, group1])
                        mean2 = np.mean(expr_filtered.loc[gene, group2])
                        
                        if mean2 != 0 and mean1 != 0:
                            log2fc = np.log2(mean1 / mean2)
                        elif mean1 > 0 and mean2 == 0:
                            log2fc = 10
                        elif mean1 == 0 and mean2 > 0:
                            log2fc = -10
                        else:
                            log2fc = 0
                        
                        if np.isfinite(log2fc) and np.isfinite(p_value):
                            diff_expr_results.append({
                                'gene': gene,
                                'log2fc': log2fc,
                                'mean_group1': mean1,
                                'mean_group2': mean2,
                                'statistic': cnv_effect,
                                'p_value': p_value,
                                'method': 'OLS'
                            })
            else:
                for gene in gene_subset:
                    if gene == driver:
                        continue
                        
                    gene_expr = expr_filtered.loc[gene, samples].values
                        
                    if np.isnan(gene_expr).any() or not np.all(np.isfinite(gene_expr)):
                        continue
                        
                    model_df = pd.DataFrame({
                        'expression': gene_expr,
                        'cnv_status': cnv_status
                    })
                        
                    for col in subtype_dummies.columns:
                        model_df[col] = subtype_dummies[col].values
                        
                    X = sm.add_constant(model_df.drop('expression', axis=1))
                    model = sm.OLS(model_df['expression'], X).fit()
                    
                    cnv_effect = model.params['cnv_status']
                    p_value = model.pvalues['cnv_status']
                    
                    mean1 = np.mean(expr_filtered.loc[gene, group1])
                    mean2 = np.mean(expr_filtered.loc[gene, group2])
                    
                    if mean2 != 0 and mean1 != 0:
                        log2fc = np.log2(mean1 / mean2)
                    elif mean1 > 0 and mean2 == 0:
                        log2fc = 10
                    elif mean1 == 0 and mean2 > 0:
                        log2fc = -10
                    else:
                        log2fc = 0
                    
                    if np.isfinite(log2fc) and np.isfinite(p_value):
                        diff_expr_results.append({
                            'gene': gene,
                            'log2fc': log2fc,
                            'mean_group1': mean1,
                            'mean_group2': mean2,
                            'statistic': cnv_effect,
                            'p_value': p_value,
                            'method': 'OLS'
                        })
        else:
            gene_subset = list(expr_filtered.index)
            if len(gene_subset) > max_genes:
                gene_subset = np.random.choice(gene_subset, max_genes, replace=False)
            
            if batch_processing:
                for i in range(0, len(gene_subset), batch_size):
                    batch_genes = gene_subset[i:i+batch_size]
                    
                    for gene in batch_genes:
                        if gene == driver:
                            continue
                            
                        group1_expr = expr_filtered.loc[gene, group1]
                        group2_expr = expr_filtered.loc[gene, group2]
                        
                        if len(group1_expr) < 3 or len(group2_expr) < 3:
                            continue
                            
                        if np.std(group1_expr) == 0 or np.std(group2_expr) == 0:
                            continue
                        
                        mean1 = np.nanmean(group1_expr)
                        mean2 = np.nanmean(group2_expr)
                        
                        if mean2 != 0 and mean1 != 0:
                            log2fc = np.log2(mean1 / mean2)
                        elif mean1 > 0 and mean2 == 0:
                            log2fc = 10
                        elif mean1 == 0 and mean2 > 0:
                            log2fc = -10
                        else:
                            log2fc = 0
                        
                        t_stat, p_value = ttest_ind(group1_expr, group2_expr, equal_var=False, nan_policy='omit')
                        
                        if np.isfinite(p_value) and np.isfinite(log2fc):
                            diff_expr_results.append({
                                'gene': gene,
                                'log2fc': log2fc,
                                'mean_group1': mean1,
                                'mean_group2': mean2,
                                'statistic': t_stat,
                                'p_value': p_value,
                                'method': 't-test'
                            })
            else:
                for gene in gene_subset:
                    if gene == driver:
                        continue
                        
                    group1_expr = expr_filtered.loc[gene, group1]
                    group2_expr = expr_filtered.loc[gene, group2]
                    
                    if len(group1_expr) < 3 or len(group2_expr) < 3:
                        continue
                        
                    if np.std(group1_expr) == 0 or np.std(group2_expr) == 0:
                        continue
                    
                    mean1 = np.nanmean(group1_expr)
                    mean2 = np.nanmean(group2_expr)
                    
                    if mean2 != 0 and mean1 != 0:
                        log2fc = np.log2(mean1 / mean2)
                    elif mean1 > 0 and mean2 == 0:
                        log2fc = 10
                    elif mean1 == 0 and mean2 > 0:
                        log2fc = -10
                    else:
                        log2fc = 0
                    
                    t_stat, p_value = ttest_ind(group1_expr, group2_expr, equal_var=False, nan_policy='omit')
                    
                    if np.isfinite(p_value) and np.isfinite(log2fc):
                        diff_expr_results.append({
                            'gene': gene,
                            'log2fc': log2fc,
                            'mean_group1': mean1,
                            'mean_group2': mean2,
                            'statistic': t_stat,
                            'p_value': p_value,
                            'method': 't-test'
                        })
        
        diff_expr_df = pd.DataFrame(diff_expr_results)
        
        if diff_expr_df.empty:
            continue
        
        diff_expr_df['abs_log2fc'] = diff_expr_df['log2fc'].abs()
        diff_expr_df = diff_expr_df.sort_values('abs_log2fc', ascending=False)
        
        _, adj_p = multipletests(diff_expr_df['p_value'], method='fdr_bh')
        diff_expr_df['adj_p_value'] = adj_p
        
        output_file = f"results/tables/transcriptional_programs/{driver}_{alteration_type.lower()}_diff_expr.tsv"
        diff_expr_df.to_csv(output_file, sep='\t', index=False)
        
        program_name = f"{driver}_{alteration_type}"
        
        plt.figure(figsize=(10, 8))
        
        is_significant = (diff_expr_df['adj_p_value'] < p_threshold) & (diff_expr_df['log2fc'].abs() > fc_threshold)
        up_mask = (diff_expr_df['log2fc'] > 0) & is_significant
        down_mask = (diff_expr_df['log2fc'] < 0) & is_significant
        nonsig_mask = ~is_significant
        
        plt.scatter(
            diff_expr_df.loc[nonsig_mask, 'log2fc'],
            -np.log10(diff_expr_df.loc[nonsig_mask, 'adj_p_value']),
            alpha=0.5, s=5, color='gray',
            label=f'Not significant ({np.sum(nonsig_mask)})'
        )
        
        if np.any(up_mask):
            plt.scatter(
                diff_expr_df.loc[up_mask, 'log2fc'],
                -np.log10(diff_expr_df.loc[up_mask, 'adj_p_value']),
                alpha=0.8, s=25, color='red',
                label=f'Up-regulated ({np.sum(up_mask)})'
            )
        
        if np.any(down_mask):
            plt.scatter(
                diff_expr_df.loc[down_mask, 'log2fc'],
                -np.log10(diff_expr_df.loc[down_mask, 'adj_p_value']),
                alpha=0.8, s=25, color='blue',
                label=f'Down-regulated ({np.sum(down_mask)})'
            )
        
        plt.axhline(y=-np.log10(p_threshold), color='gray', linestyle='--', alpha=0.7)
        plt.axvline(x=fc_threshold, color='gray', linestyle='--', alpha=0.7)
        plt.axvline(x=-fc_threshold, color='gray', linestyle='--', alpha=0.7)
        
        plt.xlabel(f'Log2 Fold Change ({group1_name} vs {group2_name})')
        plt.ylabel('-log10(adjusted p-value)')
        plt.title(f'Differential Expression Based on {driver} {alteration_type}')
        plt.legend()
        
        plt.savefig(f"results/figures/transcriptional_programs/{driver}_{alteration_type.lower()}_volcano.png", 
                    dpi=config.get("figure_dpi", 300))
        
        plt.close()
        
        up_genes = diff_expr_df[(diff_expr_df['adj_p_value'] < p_threshold) & 
                             (diff_expr_df['log2fc'] > fc_threshold)]['gene'].tolist()
        down_genes = diff_expr_df[(diff_expr_df['adj_p_value'] < p_threshold) & 
                               (diff_expr_df['log2fc'] < -fc_threshold)]['gene'].tolist()
        
        program_results[program_name] = {
            'diff_expr_df': diff_expr_df,
            'group1': group1,
            'group2': group2,
            'group1_name': group1_name,
            'group2_name': group2_name,
            'alteration_type': alteration_type,
            'up_genes': up_genes,
            'down_genes': down_genes
        }
    
    return program_results

def perform_pathway_analysis(program_results, config):
    os.makedirs("results/tables/pathways", exist_ok=True)
    os.makedirs("results/figures/pathways", exist_ok=True)
    
    pathways = {
        'PI3K-AKT-mTOR': ['PIK3CA', 'PIK3CB', 'PIK3CD', 'PIK3R1', 'AKT1', 'AKT2', 'AKT3', 'MTOR', 'PTEN', 'TSC1', 'TSC2', 'RICTOR', 'RPTOR'],
        'RAS-MAPK': ['KRAS', 'HRAS', 'NRAS', 'BRAF', 'RAF1', 'MAP2K1', 'MAP2K2', 'MAPK1', 'MAPK3', 'MAPK8', 'MAPK9', 'MAPK10'],
        'Cell Cycle': ['CCND1', 'CCNE1', 'CCNA2', 'CCNB1', 'CDK1', 'CDK2', 'CDK4', 'CDK6', 'RB1', 'E2F1', 'E2F2', 'E2F3'],
        'DNA Repair': ['BRCA1', 'BRCA2', 'PALB2', 'RAD51', 'ATM', 'ATR', 'TP53', 'CHEK1', 'CHEK2', 'MDM2'],
        'Apoptosis': ['BCL2', 'BCL2L1', 'BAX', 'BAK1', 'CASP3', 'CASP8', 'CASP9', 'BID', 'PMAIP1', 'MCL1'],
        'RTK Signaling': ['EGFR', 'ERBB2', 'ERBB3', 'ERBB4', 'FGFR1', 'FGFR2', 'FGFR3', 'FGFR4', 'IGF1R', 'MET'],
        'WNT-β-catenin': ['WNT1', 'WNT2', 'WNT3', 'WNT5A', 'FZD1', 'FZD7', 'DKK1', 'CTNNB1', 'APC', 'AXIN1', 'GSK3B'],
        'Hippo': ['YAP1', 'WWTR1', 'LATS1', 'LATS2', 'STK3', 'STK4', 'MOB1A', 'MOB1B'],
        'Notch': ['NOTCH1', 'NOTCH2', 'NOTCH3', 'NOTCH4', 'DLL1', 'DLL3', 'DLL4', 'JAG1', 'JAG2', 'HES1'],
        'Hedgehog': ['SHH', 'IHH', 'DHH', 'PTCH1', 'PTCH2', 'SMO', 'GLI1', 'GLI2', 'GLI3'],
        'Estrogen Signaling': ['ESR1', 'ESR2', 'PGR', 'FOXA1', 'GATA3', 'XBP1', 'TFF1', 'CCND1'],
        'DNA Damage Response': ['BRCA1', 'BRCA2', 'TP53', 'ATM', 'ATR', 'CHEK1', 'CHEK2', 'RAD51', 'PALB2', 'PARP1'],
        'Chromatin Remodeling': ['KMT2C', 'KMT2D', 'ARID1A', 'ARID1B', 'PBRM1', 'SMARCD1', 'SMARCA4', 'SMARCB1'],
    }
    
    hallmark_pathways = {
        'HALLMARK_E2F_TARGETS': ['E2F1', 'CDK1', 'CDC20', 'MCM2', 'RRM2', 'PCNA', 'CCNA2', 'CCNB1', 'BIRC5', 'AURKB', 'PLK1', 'MYBL2', 'CDC25A'],
        'HALLMARK_G2M_CHECKPOINT': ['CCNB1', 'CDC20', 'CDK1', 'AURKB', 'PLK1', 'BUB1', 'BIRC5', 'TOP2A', 'CCNA2', 'AURKA'],
        'HALLMARK_MYC_TARGETS_V1': ['NPM1', 'ILF2', 'HDGF', 'NOP56', 'HSPD1', 'PA2G4', 'C1QBP', 'CKS2', 'PHB', 'CCNB1'],
        'HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION': ['VIM', 'CDH2', 'SNAI1', 'SNAI2', 'TWIST1', 'ZEB1', 'ZEB2', 'FN1', 'SERPINE1', 'MMP2', 'MMP9', 'COL1A1'],
        'HALLMARK_INFLAMMATORY_RESPONSE': ['IL1A', 'IL1B', 'TNF', 'IL6', 'CXCL8', 'PTGS2', 'CCL2', 'CXCL1', 'CXCL2', 'CXCL3'],
        'HALLMARK_ESTROGEN_RESPONSE_EARLY': ['TFF1', 'PGR', 'GREB1', 'MYB', 'XBP1', 'CTSD', 'AGR2', 'PDZK1', 'ESR1', 'FOXA1'],
        'HALLMARK_PI3K_AKT_MTOR_SIGNALING': ['PIK3CA', 'PIK3CB', 'AKT1', 'AKT2', 'AKT3', 'MTOR', 'RPTOR', 'RICTOR', 'TSC1', 'TSC2', 'PTEN'],
        'HALLMARK_PROTEIN_SECRETION': ['SEC61A1', 'SSR1', 'SPCS2', 'TRAM1', 'SRPR', 'SRPRB', 'COPE', 'COPZ1', 'ARCN1', 'PREB']
    }
    
    all_pathways = {**pathways, **hallmark_pathways}
    pathway_results = {}
    
    from scipy.stats import hypergeom
    estimated_background_size = 20000
    
    for program_name, program_data in program_results.items():
        up_genes = program_data.get('up_genes', [])
        down_genes = program_data.get('down_genes', [])
        
        up_pathway_results = []
        for pathway_name, pathway_genes in all_pathways.items():
            overlap = set(up_genes) & set(pathway_genes)
            
            if len(overlap) > 0:
                overlap_count = len(overlap)
                pathway_size = len(pathway_genes)
                
                enrichment_ratio = (overlap_count / len(up_genes)) / (pathway_size / estimated_background_size) if len(up_genes) > 0 else 0
                
                p_value = hypergeom.sf(overlap_count-1, estimated_background_size, pathway_size, len(up_genes))
                
                up_pathway_results.append({
                    'pathway': pathway_name,
                    'overlap_count': overlap_count,
                    'pathway_size': pathway_size,
                    'enrichment_ratio': enrichment_ratio,
                    'overlap_genes': ', '.join(overlap),
                    'p_value': p_value
                })
        
        down_pathway_results = []
        for pathway_name, pathway_genes in all_pathways.items():
            overlap = set(down_genes) & set(pathway_genes)
            
            if len(overlap) > 0:
                overlap_count = len(overlap)
                pathway_size = len(pathway_genes)
                
                enrichment_ratio = (overlap_count / len(down_genes)) / (pathway_size / estimated_background_size) if len(down_genes) > 0 else 0
                
                p_value = hypergeom.sf(overlap_count-1, estimated_background_size, pathway_size, len(down_genes))
                
                down_pathway_results.append({
                    'pathway': pathway_name,
                    'overlap_count': overlap_count,
                    'pathway_size': pathway_size,
                    'enrichment_ratio': enrichment_ratio,
                    'overlap_genes': ', '.join(overlap),
                    'p_value': p_value
                })
        
        if up_pathway_results:
            up_pathway_df = pd.DataFrame(up_pathway_results).sort_values('p_value')
            
            if len(up_pathway_df) > 1:
                _, up_pathway_df['adj_p_value'] = multipletests(up_pathway_df['p_value'], method='fdr_bh')
            else:
                up_pathway_df['adj_p_value'] = up_pathway_df['p_value']
        else:
            up_pathway_df = pd.DataFrame(columns=['pathway', 'overlap_count', 'pathway_size', 'enrichment_ratio', 'overlap_genes', 'p_value', 'adj_p_value'])
        
        if down_pathway_results:
            down_pathway_df = pd.DataFrame(down_pathway_results).sort_values('p_value')
            
            if len(down_pathway_df) > 1:
                _, down_pathway_df['adj_p_value'] = multipletests(down_pathway_df['p_value'], method='fdr_bh')
            else:
                down_pathway_df['adj_p_value'] = down_pathway_df['p_value']
        else:
            down_pathway_df = pd.DataFrame(columns=['pathway', 'overlap_count', 'pathway_size', 'enrichment_ratio', 'overlap_genes', 'p_value', 'adj_p_value'])
        
        if not up_pathway_df.empty:
            up_pathway_df.to_csv(f"results/tables/pathways/{program_name}_up_pathway_enrichment.tsv", sep='\t', index=False)
        
        if not down_pathway_df.empty:
            down_pathway_df.to_csv(f"results/tables/pathways/{program_name}_down_pathway_enrichment.tsv", sep='\t', index=False)
            
        pathway_results[program_name] = {
            'up_pathway_df': up_pathway_df,
            'down_pathway_df': down_pathway_df,
            'method': 'Enhanced pathway analysis'
        }
    
    return pathway_results

def create_regulatory_networks(program_results, data_dict, config):
    os.makedirs("results/figures/networks", exist_ok=True)
    os.makedirs("results/tables/networks", exist_ok=True)
    
    expr_data = data_dict['expr_data']
    common_samples = list(set(data_dict['cnv_data'].columns) & set(expr_data.columns))
    initial_threshold = config.get("correlation_threshold", 0.3)
    network_layout = config.get("network_layout", "spring")
    
    network_results = {}
    
    for program_name, program_data in program_results.items():
        diff_expr_df = program_data.get('diff_expr_df')
        if diff_expr_df is None or diff_expr_df.empty:
            continue
        
        driver = program_name.split('_')[0]
        up_genes = program_data.get('up_genes', [])
        down_genes = program_data.get('down_genes', [])
        
        if not up_genes and not down_genes:
            up_genes = diff_expr_df[(diff_expr_df['adj_p_value'] < 0.05) & (diff_expr_df['log2fc'] > 1.0)]['gene'].tolist()
            down_genes = diff_expr_df[(diff_expr_df['adj_p_value'] < 0.05) & (diff_expr_df['log2fc'] < -1.0)]['gene'].tolist()
                
        signif_genes = up_genes + down_genes
        
        if not signif_genes:
            continue
        
        all_network_genes = [driver] + signif_genes
        network_genes = [g for g in all_network_genes if g in expr_data.index]
        expr_subset = expr_data.loc[network_genes, common_samples]
        
        corr_matrix = expr_subset.T.corr(method='spearman')
        
        threshold = initial_threshold
        network_edges = []
        
        while threshold >= 0.1 and len(network_edges) < 3:
            network_edges = []
            for i, gene1 in enumerate(network_genes):
                for j, gene2 in enumerate(network_genes):
                    if i < j:
                        corr = corr_matrix.loc[gene1, gene2]
                        if abs(corr) > threshold:
                            network_edges.append({
                                'source': gene1,
                                'target': gene2,
                                'correlation': corr
                            })
            if len(network_edges) >= 3:
                break
            threshold -= 0.05
        
        if len(network_edges) < 3:
            continue
            
        network = nx.Graph()
        
        for gene in network_genes:
            if gene == driver:
                node_type = 'driver'
            elif gene in up_genes:
                node_type = 'up_regulated'
            elif gene in down_genes:
                node_type = 'down_regulated'
            else:
                node_type = 'other'
            
            network.add_node(gene, type=node_type)
        
        for edge in network_edges:
            network.add_edge(
                edge['source'],
                edge['target'],
                weight=abs(edge['correlation']),
                correlation=edge['correlation']
            )
        
        nx.write_edgelist(
            network,
            f"results/tables/networks/{program_name}_network.edgelist",
            data=['weight', 'correlation']
        )
        
        plt.figure(figsize=(12, 12))
        
        if network_layout == "spring":
            pos = nx.spring_layout(network, k=1.5/np.sqrt(len(network.nodes())), seed=42, weight='weight')
        elif network_layout == "circular":
            pos = nx.circular_layout(network)
        else:
            pos = nx.spring_layout(network, k=1.5/np.sqrt(len(network.nodes())), seed=42, weight='weight')
        
        edge_weights = [abs(network[u][v]['correlation']) * 3 for u, v in network.edges()]
        edge_colors = ['red' if network[u][v]['correlation'] > 0 else 'blue' for u, v in network.edges()]
        
        nx.draw_networkx_edges(
            network, pos,
            width=edge_weights,
            alpha=0.3,
            edge_color=edge_colors
        )
        
        node_colors = {
            'driver': 'red',
            'up_regulated': 'orange',
            'down_regulated': 'blue',
            'other': 'gray'
        }
        
        for node_type, color in node_colors.items():
            nodes = [n for n, d in network.nodes(data=True) if d.get('type') == node_type]
            
            if nodes:
                size = 1000 if node_type == 'driver' else 300
                nx.draw_networkx_nodes(
                    network, pos,
                    nodelist=nodes,
                    node_color=color,
                    node_size=size,
                    alpha=0.8
                )
        
        important_nodes = [driver]
        
        if len(network.nodes()) > 5:
            degree_dict = dict(network.degree())
            sorted_degrees = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)
            for node, degree in sorted_degrees[1:5]:
                if node != driver:
                    important_nodes.append(node)
        
        nx.draw_networkx_labels(
            network, pos,
            labels={n: n for n in important_nodes},
            font_size=12,
            font_weight='bold'
        )
        
        plt.axis('off')
        plt.title(f'Gene Regulatory Network for {program_name}')
        
        plt.savefig(f"results/figures/networks/{program_name}_network.png", dpi=config.get("figure_dpi", 300))
        
        plt.close()
        
        network_results[program_name] = {
            'network': network,
            'edges': network_edges,
            'used_threshold': threshold
        }
    
    return network_results

def analyze_cnv_expression_correlations(data_dict, driver_genes, config):
    os.makedirs("results/tables/correlations", exist_ok=True)
    os.makedirs("results/figures/correlations", exist_ok=True)
    
    cnv_data = data_dict['cnv_data']
    expr_data = data_dict['expr_data']
    sample_subtypes = data_dict['sample_subtypes']
    
    common_samples = list(set(cnv_data.columns) & set(expr_data.columns))
    
    correlation_results = {}
    
    for driver in driver_genes:
        if driver not in cnv_data.index or driver not in expr_data.index:
            continue
        
        cnv_values = cnv_data.loc[driver, common_samples]
        expr_values = expr_data.loc[driver, common_samples]
        
        valid_idx = ~(np.isnan(cnv_values) | np.isnan(expr_values))
        valid_indices = np.where(valid_idx)[0]
        valid_samples = [common_samples[i] for i in valid_indices]
        
        if len(valid_samples) < 10:
            continue
        
        correlation, p_value = safe_pearsonr(cnv_values[valid_samples], expr_values[valid_samples])
        
        if np.isnan(correlation) or np.isnan(p_value):
            continue
        
        plt.figure(figsize=(10, 8))
        
        subtypes = [sample_subtypes.get(s, 'Unknown') for s in valid_samples]
        
        subtype_colors = {
            'Luminal A': '#1f77b4',
            'Luminal B': '#17becf',
            'Luminal B/HER2+': '#9467bd',
            'HER2-enriched': '#e377c2',
            'Basal-like': '#d62728',
            'Normal-like': '#2ca02c',
            'Unknown': '#7f7f7f'
        }
        
        colors = [subtype_colors.get(s, '#7f7f7f') for s in subtypes]
        
        plt.scatter(
            cnv_values[valid_samples], 
            expr_values[valid_samples],
            c=colors,
            alpha=0.7,
            s=60,
            edgecolors='black',
            linewidths=0.5
        )
        
        z = np.polyfit(cnv_values[valid_samples], expr_values[valid_samples], 1)
        p = np.poly1d(z)
        
        x_line = np.linspace(min(cnv_values[valid_samples]), max(cnv_values[valid_samples]), 100)
        y_line = p(x_line)
        
        plt.plot(x_line, y_line, "k--", alpha=0.7, linewidth=2)
        
        plt.text(
            0.05, 0.95, 
            f'Pearson r = {correlation:.3f}\np-value = {p_value:.2e}\nn = {len(valid_samples)}',
            transform=plt.gca().transAxes, 
            fontsize=11,
            verticalalignment='top', 
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray', boxstyle='round,pad=0.5')
        )
        
        plt.grid(alpha=0.3, linestyle='--')
        plt.xlabel('Copy Number Value', fontsize=12)
        plt.ylabel('Expression Level (log2)', fontsize=12)
        plt.title(f'CNV vs Expression for {driver}', fontsize=14)
        
        plt.savefig(f"results/figures/correlations/{driver}_cnv_expression.png", 
                    dpi=config.get("figure_dpi", 300))
        
        plt.close()
        
        correlation_results[driver] = {
            'correlation': correlation,
            'p_value': p_value,
            'sample_count': len(valid_samples),
            'significant': p_value < 0.05
        }
    
    summary_data = []
    for driver, result in correlation_results.items():
        summary_entry = {
            'gene': driver,
            'correlation': result['correlation'],
            'p_value': result['p_value'],
            'sample_count': result['sample_count'],
            'significant': result['significant']
        }
        
        summary_data.append(summary_entry)
    
    summary_df = pd.DataFrame(summary_data).sort_values('correlation', ascending=False)
    summary_df.to_csv("results/tables/correlations/cnv_expression_correlation_summary.tsv", sep='\t', index=False)
    
    return correlation_results, summary_df

def analyze_survival_impact(program_results, data_dict, config):
    if 'survival_data' not in data_dict:
        return None
    
    survival_data = data_dict['survival_data']
    
    required_cols = ['sample', 'OS', 'OS.time']
    if not all(col in survival_data.columns for col in required_cols):
        return None
    
    os.makedirs("results/figures/survival", exist_ok=True)
    os.makedirs("results/tables/survival", exist_ok=True)
    
    expr_data = data_dict['expr_data']
    cnv_data = data_dict['cnv_data']
    
    survival_results = {}
    
    for program_name, program_data in program_results.items():
        parts = program_name.split('_')
        driver = parts[0]
        alteration_type = parts[1] if len(parts) > 1 else "alteration"
        
        up_genes = program_data.get('up_genes', [])
        down_genes = program_data.get('down_genes', [])
        
        min_genes = config.get("min_genes_per_program", 10)
        if len(up_genes) + len(down_genes) < min_genes:
            continue
        
        signature_genes = up_genes + down_genes
        signature_genes = [g for g in signature_genes if g in expr_data.index]
        
        if len(signature_genes) < min_genes:
            continue
        
        common_samples = list(set(expr_data.columns) & set(survival_data['sample']))
        
        if len(common_samples) < 30:
            continue
        
        expr_zscore = pd.DataFrame(
            StandardScaler().fit_transform(expr_data.loc[signature_genes, common_samples].T).T,
            index=signature_genes,
            columns=common_samples
        )
        
        signature_scores = pd.Series(0, index=common_samples)
        
        for gene in up_genes:
            if gene in expr_zscore.index:
                signature_scores += expr_zscore.loc[gene]
        
        for gene in down_genes:
            if gene in expr_zscore.index:
                signature_scores -= expr_zscore.loc[gene]
        
        num_genes = len([g for g in up_genes if g in expr_zscore.index]) + len([g for g in down_genes if g in expr_zscore.index])
        if num_genes > 0:
            signature_scores /= num_genes
        
        survival_df = survival_data[survival_data['sample'].isin(common_samples)].copy()
        survival_df['signature_score'] = [signature_scores[s] for s in survival_df['sample']]
        
        median_score = survival_df['signature_score'].median()
        survival_df['signature_group'] = ['High' if s > median_score else 'Low' for s in survival_df['signature_score']]
        
        if driver in cnv_data.index:
            driver_cnv = cnv_data.loc[driver]
            if alteration_type.lower() == 'amplification':
                survival_df['cnv_status'] = ['Amplified' if driver_cnv.get(s, 0) >= 1 else 'Not amplified' 
                                            for s in survival_df['sample']]
            else:
                survival_df['cnv_status'] = ['Deleted' if driver_cnv.get(s, 0) <= -1 else 'Not deleted' 
                                            for s in survival_df['sample']]
        
        kmf = KaplanMeierFitter()
        
        plt.figure(figsize=(10, 8))
        
        high_df = survival_df[survival_df['signature_group'] == 'High']
        low_df = survival_df[survival_df['signature_group'] == 'Low']
        
        kmf.fit(high_df['OS.time'], high_df['OS'], label=f'High Signature (n={len(high_df)})')
        high_curve = kmf.plot(ci_show=True, color='red', alpha=0.7)
        
        kmf.fit(low_df['OS.time'], low_df['OS'], label=f'Low Signature (n={len(low_df)})')
        low_curve = kmf.plot(ci_show=True, color='blue', alpha=0.7)
        
        results = logrank_test(high_df['OS.time'], low_df['OS.time'], 
                              high_df['OS'], low_df['OS'])
        
        plt.text(0.05, 0.05, f'Logrank p-value: {results.p_value:.3e}', 
                transform=plt.gca().transAxes, fontsize=12,
                bbox=dict(facecolor='white', alpha=0.7))
        
        plt.title(f'Survival Analysis - {program_name} Signature')
        plt.xlabel('Time (Days)')
        plt.ylabel('Survival Probability')
        plt.grid(alpha=0.3)
        plt.ylim(0, 1.05)
        
        plt.savefig(f"results/figures/survival/{program_name}_survival.png", 
                    dpi=config.get("figure_dpi", 300))
        
        plt.close()
        
        if len(survival_df) >= 50:
            cox_df = survival_df[['OS.time', 'OS', 'signature_score']].copy()
            
            if 'cnv_status' in survival_df.columns:
                cox_df['cnv_altered'] = (survival_df['cnv_status'] == 'Amplified').astype(int) if alteration_type.lower() == 'amplification' else (survival_df['cnv_status'] == 'Deleted').astype(int)
            
            clinical_covariates = ['age', 'stage', 'grade', 'tumor_size', 'lymph_node_status']
            available_covariates = [col for col in clinical_covariates if col in survival_df.columns]
            
            if available_covariates:
                for col in available_covariates:
                    cox_df[col] = survival_df[col]
            
            cph = CoxPHFitter()
            cph.fit(cox_df, duration_col='OS.time', event_col='OS')
            
            with open(f"results/tables/survival/{program_name}_cox_summary.txt", 'w') as f:
                f.write(str(cph.summary))
            
            cox_summary = cph.summary.reset_index()
            cox_summary.columns = ['variable' if col == 'index' else col for col in cox_summary.columns]
            cox_summary.to_csv(f"results/tables/survival/{program_name}_cox_results.tsv", sep='\t', index=False)
            
            survival_results[program_name] = {
                'logrank_p_value': results.p_value,
                'cox_model': {
                    var: {
                        'hazard_ratio': np.exp(coef),
                        'p_value': p
                    }
                    for var, coef, p in zip(
                        cox_summary['variable'],
                        cox_summary['coef'],
                        cox_summary['p']
                    )
                }
            }
        else:
            survival_results[program_name] = {
                'logrank_p_value': results.p_value
            }
    
    return survival_results

def analyze_subtype_specific_programs(data_dict, program_results, config):
    os.makedirs("results/figures/subtype_programs", exist_ok=True)
    os.makedirs("results/tables/subtype_programs", exist_ok=True)
    
    cnv_data = data_dict['cnv_data']
    expr_data = data_dict['expr_data']
    sample_subtypes = data_dict['sample_subtypes']
    
    common_samples = list(set(cnv_data.columns) & set(expr_data.columns))
    common_samples = [s for s in common_samples if s in sample_subtypes]
    
    subtype_samples = {}
    for subtype in set(sample_subtypes.values()):
        subtype_samples[subtype] = [s for s in common_samples if sample_subtypes[s] == subtype]
    
    min_samples = config.get("min_samples_per_group", 5)
    
    subtype_samples = {k: v for k, v in subtype_samples.items() if len(v) >= min_samples}
    
    subtype_program_results = {}
    
    for program_name, program_data in program_results.items():
        diff_expr_df = program_data['diff_expr_df']
        group1 = program_data['group1']
        group2 = program_data['group2']
        alteration_type = program_data['alteration_type']
        
        driver = program_name.split('_')[0]
        
        driver_cnv = cnv_data.loc[driver, common_samples]
        
        subtype_results = {}
        
        for subtype, samples in subtype_samples.items():
            subtype_altered = [s for s in samples if s in group1]
            subtype_neutral = [s for s in samples if s in group2]
            
            if len(subtype_altered) < min_samples or len(subtype_neutral) < min_samples:
                continue
                
            subtype_diff_expr = []
            
            genes_to_test = diff_expr_df.sort_values('abs_log2fc', ascending=False).head(1000)['gene'].tolist()
            
            for gene in genes_to_test:
                if gene == driver:
                    continue
                    
                alt_expr = expr_data.loc[gene, subtype_altered]
                neut_expr = expr_data.loc[gene, subtype_neutral]
                
                alt_mean = alt_expr.mean()
                neut_mean = neut_expr.mean()
                
                if alt_mean > 0 and neut_mean > 0:
                    log2fc = np.log2(alt_mean / neut_mean)
                elif alt_mean > 0 and neut_mean == 0:
                    log2fc = 10
                elif alt_mean == 0 and neut_mean > 0:
                    log2fc = -10
                else:
                    log2fc = 0
                
                if (len(alt_expr) >= 3 and len(neut_expr) >= 3 and 
                    np.std(alt_expr) > 0 and np.std(neut_expr) > 0):
                    t_stat, p_value = ttest_ind(alt_expr, neut_expr, equal_var=False)
                    
                    subtype_diff_expr.append({
                        'gene': gene,
                        'log2fc': log2fc,
                        'p_value': p_value,
                        'mean_altered': alt_mean,
                        'mean_neutral': neut_mean
                    })
            
            subtype_diff_df = pd.DataFrame(subtype_diff_expr)
            
            if subtype_diff_df.empty:
                continue
            
            subtype_diff_df['abs_log2fc'] = subtype_diff_df['log2fc'].abs()
            
            _, subtype_diff_df['adj_p_value'] = multipletests(subtype_diff_df['p_value'], method='fdr_bh')
            
            subtype_diff_df.to_csv(
                f"results/tables/subtype_programs/{program_name}_{subtype.replace('/', '_')}_diff_expr.tsv", 
                sep='\t', 
                index=False
            )
            
            overall_fc = diff_expr_df[diff_expr_df['gene'].isin(subtype_diff_df['gene'])].set_index('gene')['log2fc']
            subtype_fc = subtype_diff_df.set_index('gene')['log2fc']
            
            common_genes = list(set(overall_fc.index) & set(subtype_fc.index))
            
            subtype_results[subtype] = {
                'diff_expr_df': subtype_diff_df,
                'common_genes': common_genes
            }
            
            if len(common_genes) > 3:
                corr, p_val = safe_pearsonr(
                    [overall_fc.loc[g] for g in common_genes],
                    [subtype_fc.loc[g] for g in common_genes]
                )
                subtype_results[subtype]['correlation'] = corr
                subtype_results[subtype]['correlation_p_value'] = p_val
        
        subtype_program_results[program_name] = subtype_results
    
    return subtype_program_results

def match_genes_flexibly(program_genes, external_genes):
    external_gene_list = list(external_genes)
    
    common_genes = list(set(program_genes).intersection(set(external_gene_list)))
    
    if len(common_genes) < 3:
        prog_lower = {g.lower(): g for g in program_genes}
        ext_lower = {g.lower(): g for g in external_gene_list}
        
        common_lower = set(prog_lower.keys()).intersection(set(ext_lower.keys()))
        
        for lower_gene in common_lower:
            original_gene = prog_lower[lower_gene]
            if original_gene not in common_genes:
                common_genes.append(original_gene)
    
    if len(common_genes) < 3:
        for prog_gene in program_genes:
            if prog_gene in common_genes:
                continue
                
            for ext_gene in external_gene_list:
                if ext_gene.startswith(prog_gene) and len(prog_gene) >= 3:
                    common_genes.append(prog_gene)
                    break
    
    return common_genes

def validate_transcriptional_programs(program_results, external_data, data_dict, config):
    if external_data is None or 'expression' not in external_data:
        return None
    
    os.makedirs("results/figures/validation", exist_ok=True)
    os.makedirs("results/tables/validation", exist_ok=True)
    
    external_expr = external_data['expression']
    external_cnv = external_data.get('cnv', None)
    
    external_subtypes = external_data.get('sample_subtypes', {})
    
    validation_min_genes = config.get("validation_min_genes", 2)
    
    validation_results = {}
    
    for program_name, program_data in program_results.items():
        up_genes = program_data.get('up_genes', [])
        down_genes = program_data.get('down_genes', [])
        
        common_up = match_genes_flexibly(up_genes, external_expr.index)
        common_down = match_genes_flexibly(down_genes, external_expr.index)
        
        if len(common_up) < validation_min_genes and len(common_down) < validation_min_genes:
            continue
        
        validation_score = (len(common_up) + len(common_down)) / (len(up_genes) + len(down_genes) + 0.1)
        
        driver = program_name.split('_')[0]
        alteration_type = program_name.split('_')[1] if len(program_name.split('_')) > 1 else 'Alteration'
        
        if external_cnv is not None and driver in external_cnv.index:
            common_ext_samples = list(set(external_cnv.columns) & set(external_expr.columns))
            
            driver_cnv = external_cnv.loc[driver, common_ext_samples]
            
            if alteration_type.lower() == 'amplification':
                ext_altered_samples = [s for s in common_ext_samples if driver_cnv[s] >= 1]
                ext_neutral_samples = [s for s in common_ext_samples if -1 < driver_cnv[s] < 1]
            else:
                ext_altered_samples = [s for s in common_ext_samples if driver_cnv[s] <= -1]
                ext_neutral_samples = [s for s in common_ext_samples if -1 < driver_cnv[s] < 1]
            
            min_samples = config.get("min_samples_per_group", 5)
            
            if len(ext_altered_samples) >= min_samples and len(ext_neutral_samples) >= min_samples:
                ext_diff_expr_results = []
                
                test_genes = common_up + common_down
                
                for gene in test_genes:
                    altered_expr = external_expr.loc[gene, ext_altered_samples]
                    neutral_expr = external_expr.loc[gene, ext_neutral_samples]
                    
                    if (len(altered_expr) < 3 or len(neutral_expr) < 3 or 
                        np.std(altered_expr) == 0 or np.std(neutral_expr) == 0):
                        continue
                    
                    altered_mean = altered_expr.mean()
                    neutral_mean = neutral_expr.mean()
                    
                    if altered_mean > 0 and neutral_mean > 0:
                        log2fc = np.log2(altered_mean / neutral_mean)
                    elif altered_mean > 0 and neutral_mean == 0:
                        log2fc = 10
                    elif altered_mean == 0 and neutral_mean > 0:
                        log2fc = -10
                    else:
                        log2fc = 0
                    
                    t_stat, p_value = ttest_ind(altered_expr, neutral_expr, equal_var=False)
                    
                    ext_diff_expr_results.append({
                        'gene': gene,
                        'log2fc': log2fc,
                        'p_value': p_value,
                        'expected_direction': gene in common_up if log2fc > 0 else gene in common_down
                    })
                
                ext_diff_df = pd.DataFrame(ext_diff_expr_results)
                
                if not ext_diff_df.empty:
                    correct_direction = ext_diff_df['expected_direction'].sum()
                    total_genes = len(ext_diff_df)
                    agreement_score = correct_direction / total_genes if total_genes > 0 else 0
                    
                    _, ext_diff_df['adj_p_value'] = multipletests(ext_diff_df['p_value'], method='fdr_bh')
                    
                    ext_diff_df.to_csv(f"results/tables/validation/{program_name}_validation.tsv", sep='\t', index=False)
                    
                    orig_diff_df = program_data['diff_expr_df']
                    common_genes = set(ext_diff_df['gene']) & set(orig_diff_df['gene'])
                    
                    plot_data = pd.DataFrame(index=common_genes)
                    plot_data['original_fc'] = orig_diff_df.set_index('gene').loc[common_genes, 'log2fc']
                    plot_data['validation_fc'] = ext_diff_df.set_index('gene').loc[common_genes, 'log2fc']
                    
                    corr, p_val = safe_pearsonr(plot_data['original_fc'], plot_data['validation_fc'])
                    
                    validation_results[program_name] = {
                        'common_up': common_up,
                        'common_down': common_down,
                        'validation_score': validation_score,
                        'agreement_score': agreement_score,
                        'correlation': corr,
                        'correlation_p_value': p_val,
                        'external_diff_expr': ext_diff_df,
                        'original_diff_expr': orig_diff_df
                    }
                else:
                    validation_results[program_name] = {
                        'common_up': common_up,
                        'common_down': common_down,
                        'validation_score': validation_score,
                        'agreement_score': None,
                        'correlation': None,
                        'correlation_p_value': None
                    }
            else:
                validation_results[program_name] = {
                    'common_up': common_up,
                    'common_down': common_down,
                    'validation_score': validation_score
                }
        else:
            validation_results[program_name] = {
                'common_up': common_up,
                'common_down': common_down,
                'validation_score': validation_score
            }
    
    if validation_results:
        summary_data = []
        
        for program, results in validation_results.items():
            summary_entry = {
                'program': program,
                'common_up_genes': len(results['common_up']),
                'common_down_genes': len(results['common_down']),
                'validation_score': results['validation_score']
            }
            
            if 'agreement_score' in results and results['agreement_score'] is not None:
                summary_entry['agreement_score'] = results['agreement_score']
                
            if 'correlation' in results and results['correlation'] is not None:
                summary_entry['correlation'] = results['correlation']
                summary_entry['correlation_p_value'] = results['correlation_p_value']
            
            summary_data.append(summary_entry)
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv("results/tables/validation/validation_summary.tsv", sep='\t', index=False)
    
    return validation_results

def run_transcriptional_program_analysis():
    start_time = time.time()
    
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)
    
    config = load_config()
    data_dict = load_processed_data()
    
    if config.get("use_external_validation", True):
        external_data = load_external_validation_data()
    else:
        external_data = None
    
    if 'driver_df' in data_dict:
        driver_df = data_dict['driver_df']
        
        gene_col = None
        for possible_name in ['gene', 'Gene', 'symbol', 'Symbol', 'hugo_symbol', 'gene_symbol', 'GeneSymbol']:
            if possible_name in driver_df.columns:
                gene_col = possible_name
                break
        
        score_col = None
        for possible_name in ['evidence_score', 'score', 'Score', 'rank_score', 'driver_score']:
            if possible_name in driver_df.columns:
                score_col = possible_name
                break
        
        if gene_col and score_col:
            driver_genes = list(driver_df.sort_values(score_col, ascending=False).head(20)[gene_col])
        else:
            if driver_df.index.name in ['gene', 'Gene', 'symbol', 'Symbol'] or isinstance(driver_df.index[0], str):
                if score_col:
                    driver_genes = list(driver_df.sort_values(score_col, ascending=False).head(20).index)
                else:
                    driver_genes = list(driver_df.head(20).index)
            else:
                driver_genes = ['ERBB2', 'MYC', 'CCND1', 'EGFR', 'PIK3CA', 'PTEN', 'TP53', 'RB1']
    else:
        driver_genes = ['ERBB2', 'MYC', 'CCND1', 'EGFR', 'PIK3CA', 'PTEN', 'TP53', 'RB1']
    
    correlation_results, correlation_summary = analyze_cnv_expression_correlations(
        data_dict,
        driver_genes,
        config
    )
    
    program_results = identify_transcriptional_programs(data_dict, config)
    
    if not program_results:
        return {
            'success': False,
            'error': 'No transcriptional programs identified'
        }
    
    pathway_results = perform_pathway_analysis(program_results, config)
    network_results = create_regulatory_networks(program_results, data_dict, config)
    subtype_program_results = analyze_subtype_specific_programs(data_dict, program_results, config)
    
    if external_data:
        validation_results = validate_transcriptional_programs(
            program_results, external_data, data_dict, config
        )
    else:
        validation_results = None
    
    if config.get("use_clinical_integration", False):
        survival_results = analyze_survival_impact(program_results, data_dict, config)
    else:
        survival_results = None
    
    summary = {
        'total_programs': len(program_results),
        'total_pathways': len(pathway_results) if pathway_results else 0,
        'total_networks': len(network_results) if network_results else 0,
        'subtype_programs': len(subtype_program_results) if subtype_program_results else 0,
        'validated_programs': len(validation_results) if validation_results else 0,
        'significant_survival': sum(1 for res in survival_results.values() if res['logrank_p_value'] < 0.05) if survival_results else 0,
        'execution_time': time.time() - start_time
    }
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv("results/transcriptional_programs_summary.tsv", sep='\t', index=False)
    
    return {
        'success': True,
        'program_results': program_results,
        'pathway_results': pathway_results,
        'network_results': network_results,
        'subtype_program_results': subtype_program_results,
        'validation_results': validation_results,
        'survival_results': survival_results,
        'summary': summary
    }

if __name__ == "__main__":
    result = run_transcriptional_program_analysis()
