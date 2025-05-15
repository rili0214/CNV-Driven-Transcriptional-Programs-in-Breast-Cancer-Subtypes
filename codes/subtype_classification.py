import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import umap.umap_ as umap
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

def perform_consensus_clustering(data, n_iter=150, k_range=range(2, 9)):
    consensus_matrices = {}
    connectivity_matrices = {}
    sample_occurrence = np.zeros(data.shape[0])

    for k in k_range:
        consensus_matrices[k] = np.zeros((data.shape[0], data.shape[0]))
        connectivity_matrices[k] = np.zeros((data.shape[0], data.shape[0]))

    for i in range(n_iter):
        subsample_size = int(0.8 * data.shape[0])
        subsample_idx = np.random.choice(data.shape[0], size=subsample_size, replace=False)
        subsample_data = data[subsample_idx]
        sample_occurrence[subsample_idx] += 1

        Z = linkage(subsample_data, method='ward')

        for k in k_range:
            labels = fcluster(Z, k, criterion='maxclust')
            for i1, idx1 in enumerate(subsample_idx):
                for i2, idx2 in enumerate(subsample_idx):
                    connectivity_matrices[k][idx1, idx2] += 1
                    if labels[i1] == labels[i2]:
                        consensus_matrices[k][idx1, idx2] += 1
    
    for k in k_range:
        consensus_matrices[k] = np.divide(
            consensus_matrices[k], 
            connectivity_matrices[k], 
            out=np.zeros_like(consensus_matrices[k]), 
            where=connectivity_matrices[k]!=0
        )
    
    return consensus_matrices, connectivity_matrices, sample_occurrence

def evaluate_clustering(consensus_matrices):
    metrics = {}
    
    for k, matrix in consensus_matrices.items():
        pac = np.sum((matrix > 0.1) & (matrix < 0.9)) / (matrix.shape[0] * matrix.shape[1])
        if k > min(consensus_matrices.keys()):
            prev_k = k - 1
            prev_matrix = consensus_matrices[prev_k]

            x_curr = np.sort(matrix.flatten())
            y_curr = np.arange(1, len(x_curr) + 1) / len(x_curr)
            
            x_prev = np.sort(prev_matrix.flatten())
            y_prev = np.arange(1, len(x_prev) + 1) / len(x_prev)

            area_curr = np.trapz(y_curr, x_curr)
            area_prev = np.trapz(y_prev, x_prev)

            rel_change = (area_curr - area_prev) / area_prev
        else:
            rel_change = None

        distance_matrix = 1 - matrix

        Z = linkage(distance_matrix[np.triu_indices(distance_matrix.shape[0], k=1)], method='average')
        labels = fcluster(Z, k, criterion='maxclust')
        sil_score = silhouette_score(distance_matrix, labels, metric='precomputed')
        sil_samples = silhouette_samples(distance_matrix, labels, metric='precomputed')

        cluster_sil = {}
        for i in range(1, k+1):
            cluster_indices = np.where(labels == i)[0]
            if len(cluster_indices) > 0:
                cluster_sil[i] = np.mean(sil_samples[cluster_indices])

        entropy = 0
        for i in range(1, k+1):
            cluster_indices = np.where(labels == i)[0]
            if len(cluster_indices) > 1:
                submatrix = matrix[np.ix_(cluster_indices, cluster_indices)]
                submatrix = np.clip(submatrix, 0.001, 0.999)
                entropy -= np.sum(submatrix * np.log2(submatrix) + 
                                  (1-submatrix) * np.log2(1-submatrix))

        entropy /= (matrix.shape[0] * matrix.shape[1])
            
        separation = 0
        for i in range(1, k+1):
            for j in range(i+1, k+1):
                idx_i = np.where(labels == i)[0]
                idx_j = np.where(labels == j)[0]
                if len(idx_i) > 0 and len(idx_j) > 0:
                    separation += np.mean(1 - matrix[np.ix_(idx_i, idx_j)])
            
        if k > 1: 
            separation /= (k * (k - 1) / 2) 

        metrics[k] = {
            'pac': pac,
            'rel_change': rel_change,
            'silhouette': sil_score,
            'cluster_silhouette': cluster_sil,
            'sample_silhouette': dict(zip(range(len(sil_samples)), sil_samples)),
            'entropy': entropy,
            'separation': separation if k > 1 else 0,
            'labels': labels
        }
        
    best_k = find_optimal_k(metrics)
    return metrics, best_k

def get_cluster_assignments(consensus_matrix, k, sample_names=None):
    distance_matrix = 1 - consensus_matrix
    Z = linkage(distance_matrix[np.triu_indices(distance_matrix.shape[0], k=1)], method='average')
    labels = fcluster(Z, k, criterion='maxclust')
    
    sil_score = silhouette_score(distance_matrix, labels, metric='precomputed')
    sample_silhouette_values = silhouette_samples(distance_matrix, labels, metric='precomputed')
    confidence_scores = calculate_cluster_confidence(consensus_matrix, labels)
    
    if sample_names is None:
        sample_names = [f"Sample_{i}" for i in range(len(labels))]
        
    combined_confidence = (sample_silhouette_values + 1) / 2 * 0.5 + confidence_scores * 0.5
    confidence_categories = np.array(['Low'] * len(combined_confidence), dtype=object)
    confidence_categories[combined_confidence >= 0.5] = 'Medium'
    confidence_categories[combined_confidence >= 0.75] = 'High'
    
    result_df = pd.DataFrame({
        'sample': sample_names,
        'cluster': labels,
        'silhouette': sample_silhouette_values,
        'consensus_confidence': confidence_scores,
        'combined_confidence': combined_confidence,
        'confidence_level': confidence_categories
    })
    
    os.makedirs("results/tables/subtypes", exist_ok=True)
    result_df.to_csv(f"results/tables/subtypes/cluster_assignments_k{k}.tsv", sep='\t', index=False)
    
    return result_df, sample_silhouette_values, confidence_scores

def calculate_cluster_confidence(consensus_matrix, labels):
    n_samples = len(labels)
    confidence_scores = np.zeros(n_samples)
    
    for i in range(n_samples):
        same_cluster_indices = np.where(labels == labels[i])[0]
        within_cluster_consensus = np.mean(consensus_matrix[i, same_cluster_indices])
        
        different_cluster_indices = np.where(labels != labels[i])[0]
        between_cluster_consensus = np.mean(consensus_matrix[i, different_cluster_indices]) if len(different_cluster_indices) > 0 else 0
        
        confidence = (within_cluster_consensus - between_cluster_consensus + 1) / 2
        confidence_scores[i] = confidence
    
    return confidence_scores

def characterize_clusters(integrated_features, labels, k):
    os.makedirs("results/figures/subtypes", exist_ok=True)
    
    labeled_data = integrated_features.copy()
    cluster_labels = labels.tolist() if hasattr(labels, 'tolist') else list(labels)
    labeled_data['cluster'] = cluster_labels
    
    cluster_means = labeled_data.groupby('cluster').mean()
    cluster_means = cluster_means.fillna(0)
    
    return cluster_means

def assign_subtypes(cluster_means, integrated_features, cluster_assignments_df, reference_profiles=None):
    er_markers = [col for col in cluster_means.columns if 'ESR1' in col or 'ER' in col]
    pr_markers = [col for col in cluster_means.columns if 'PGR' in col or 'PR' in col]
    her2_markers = [col for col in cluster_means.columns if 'ERBB2' in col or 'HER2' in col]
    proliferation_markers = [col for col in cluster_means.columns if 'MKI67' in col or 'Ki67' in col]
    basal_markers = [col for col in cluster_means.columns if 'KRT5' in col or 'KRT14' in col]

    cluster_scores = {}
    for cluster in cluster_means.index:
        er_score = cluster_means.loc[cluster, er_markers].mean() if er_markers else 0
        pr_score = cluster_means.loc[cluster, pr_markers].mean() if pr_markers else 0
        her2_score = cluster_means.loc[cluster, her2_markers].mean() if her2_markers else 0
        prolif_score = cluster_means.loc[cluster, proliferation_markers].mean() if proliferation_markers else 0
        basal_score = cluster_means.loc[cluster, basal_markers].mean() if basal_markers else 0
        
        cluster_scores[cluster] = {
            'ER': er_score,
            'PR': pr_score,
            'HER2': her2_score,
            'Proliferation': prolif_score,
            'Basal': basal_score
        }
    
    score_df = pd.DataFrame(cluster_scores).T
    z_scores = score_df.apply(lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0)

    if reference_profiles is None:
        reference_profiles = {
            'Luminal A': {'ER': 'high', 'PR': 'high', 'HER2': 'low', 'Proliferation': 'low', 'Basal': 'low'},
            'Luminal B': {'ER': 'high', 'PR': 'variable', 'HER2': 'low', 'Proliferation': 'high', 'Basal': 'low'},
            'Luminal B/HER2+': {'ER': 'high', 'PR': 'variable', 'HER2': 'high', 'Proliferation': 'high', 'Basal': 'low'},
            'HER2-enriched': {'ER': 'low', 'PR': 'low', 'HER2': 'high', 'Proliferation': 'high', 'Basal': 'low'},
            'Basal-like': {'ER': 'low', 'PR': 'low', 'HER2': 'low', 'Proliferation': 'high', 'Basal': 'high'}
        }

    subtype_map = {}
    subtype_confidence = {}
    
    for cluster in z_scores.index:
        er_z = z_scores.loc[cluster, 'ER']
        pr_z = z_scores.loc[cluster, 'PR']
        her2_z = z_scores.loc[cluster, 'HER2']
        prolif_z = z_scores.loc[cluster, 'Proliferation']
        basal_z = z_scores.loc[cluster, 'Basal']
        
        subtype_similarity = {}
        for subtype, profile in reference_profiles.items():
            similarity = 0
            evidence = []
            
            if profile['ER'] == 'high' and er_z > 0.5:
                similarity += 2
                evidence.append(f"ER+ (z={er_z:.2f})")
            elif profile['ER'] == 'low' and er_z < -0.5:
                similarity += 2
                evidence.append(f"ER- (z={er_z:.2f})")
            elif profile['ER'] == 'medium' and abs(er_z) <= 0.5:
                similarity += 1
                evidence.append(f"ER medium (z={er_z:.2f})")
            
            if profile['PR'] == 'high' and pr_z > 0.5:
                similarity += 1.5
                evidence.append(f"PR+ (z={pr_z:.2f})")
            elif profile['PR'] == 'low' and pr_z < -0.5:
                similarity += 1.5
                evidence.append(f"PR- (z={pr_z:.2f})")
            elif profile['PR'] == 'medium' and abs(pr_z) <= 0.5:
                similarity += 0.75
                evidence.append(f"PR medium (z={pr_z:.2f})")
            elif profile['PR'] == 'variable':
                similarity += 0.5
            
            if profile['HER2'] == 'high' and her2_z > 0.5:
                similarity += 2
                evidence.append(f"HER2+ (z={her2_z:.2f})")
            elif profile['HER2'] == 'low' and her2_z < 0.5:
                similarity += 0.5
                evidence.append(f"HER2- (z={her2_z:.2f})")
            
            if profile['Proliferation'] == 'high' and prolif_z > 0.5:
                similarity += 1
                evidence.append(f"High proliferation (z={prolif_z:.2f})")
            elif profile['Proliferation'] == 'low' and prolif_z < 0:
                similarity += 1
                evidence.append(f"Low proliferation (z={prolif_z:.2f})")
            
            if profile['Basal'] == 'high' and basal_z > 0.5:
                similarity += 1.5
                evidence.append(f"Basal+ (z={basal_z:.2f})")
            elif profile['Basal'] == 'low' and basal_z < 0:
                similarity += 0.5
                evidence.append(f"Basal- (z={basal_z:.2f})")
            elif profile['Basal'] == 'medium' and abs(basal_z) <= 0.5:
                similarity += 0.5
                evidence.append(f"Basal medium (z={basal_z:.2f})")
            
            subtype_similarity[subtype] = {'score': similarity, 'evidence': evidence}
        
        best_subtype = max(subtype_similarity.items(), key=lambda x: x[1]['score'])
        subtype_map[cluster] = best_subtype[0]
        
        best_score = best_subtype[1]['score']
        max_possible_score = 8
        confidence = best_score / max_possible_score
        
        scores = [item[1]['score'] for item in subtype_similarity.items()]
        if len(scores) > 1:
            scores.sort(reverse=True)
            best_second_gap = (scores[0] - scores[1]) / max(scores[0], 1e-6)
            confidence = 0.7 * confidence + 0.3 * best_second_gap
        
        subtype_confidence[cluster] = {
            'confidence': confidence,
            'evidence': best_subtype[1]['evidence'],
            'similarity_scores': {k: v['score'] for k, v in subtype_similarity.items()}
        }
    
    for cluster in cluster_assignments_df['cluster'].unique():
        if cluster not in subtype_map:
            default_subtypes = ['Luminal A', 'Luminal B', 'Basal-like', 'HER2-enriched', 'Normal-like']
            idx = min(int(cluster) - 1, len(default_subtypes) - 1)
            subtype_map[cluster] = default_subtypes[idx]
            
            subtype_confidence[cluster] = {
                'confidence': 0.5,
                'evidence': ['Default assignment'],
                'similarity_scores': {s: 0 for s in default_subtypes}
            }
            subtype_confidence[cluster]['similarity_scores'][subtype_map[cluster]] = 0.5
    
    sample_subtypes = pd.DataFrame({
        'sample': cluster_assignments_df['sample'],
        'cluster': cluster_assignments_df['cluster'],
        'subtype': [subtype_map[c] for c in cluster_assignments_df['cluster']],
        'cluster_confidence': cluster_assignments_df['combined_confidence'],
        'subtype_confidence': [subtype_confidence[c]['confidence'] for c in cluster_assignments_df['cluster']],
        'combined_confidence': np.zeros(len(cluster_assignments_df))
    })
    
    for i, row in sample_subtypes.iterrows():
        sample_subtypes.loc[i, 'combined_confidence'] = row['cluster_confidence'] * row['subtype_confidence']
    
    sample_subtypes['confidence_level'] = 'Medium'
    sample_subtypes.loc[sample_subtypes['combined_confidence'] < 0.4, 'confidence_level'] = 'Low'
    sample_subtypes.loc[sample_subtypes['combined_confidence'] >= 0.6, 'confidence_level'] = 'High'
    
    return sample_subtypes, subtype_map, z_scores, subtype_confidence

def find_optimal_k(metrics):
    k_values = sorted(metrics.keys())
    scores = {}
    
    weights = {
        'silhouette': 0.4,
        'pac': 0.3,
        'rel_change': 0.2,
        'entropy': 0.1
    }
    
    for metric_name, weight in weights.items():
        if metric_name == 'silhouette':
            silhouette_scores = {k: metrics[k]['silhouette'] for k in k_values 
                               if 'silhouette' in metrics[k] and not np.isnan(metrics[k]['silhouette'])}
            
            if silhouette_scores:
                max_silhouette = max(silhouette_scores.values())
                min_silhouette = min(silhouette_scores.values())
                range_silhouette = max_silhouette - min_silhouette
                
                for k in silhouette_scores:
                    if k not in scores:
                        scores[k] = 0
                    if range_silhouette > 0:
                        norm_score = (silhouette_scores[k] - min_silhouette) / range_silhouette
                    else:
                        norm_score = 1 if silhouette_scores[k] == max_silhouette else 0
                    scores[k] += weights['silhouette'] * norm_score
        
        elif metric_name == 'pac':
            pac_values = {k: metrics[k]['pac'] for k in k_values}
            
            if pac_values:
                max_pac = max(pac_values.values())
                min_pac = min(pac_values.values())
                range_pac = max_pac - min_pac
                
                for k in pac_values:
                    if k not in scores:
                        scores[k] = 0
                    if range_pac > 0:
                        norm_score = 1 - (pac_values[k] - min_pac) / range_pac
                    else:
                        norm_score = 1 if pac_values[k] == min_pac else 0
                    scores[k] += weights['pac'] * norm_score
        
        elif metric_name == 'rel_change':
            rel_changes = {k: metrics[k]['rel_change'] for k in k_values if metrics[k]['rel_change'] is not None}
            
            if rel_changes:
                k_values_rc = sorted(rel_changes.keys())
                drops = []
                
                for i in range(1, len(k_values_rc)):
                    current_k = k_values_rc[i]
                    prev_k = k_values_rc[i-1]
                    if current_k in rel_changes and prev_k in rel_changes:
                        drop = rel_changes[prev_k] - rel_changes[current_k]
                        drops.append((current_k, drop))
                
                if drops:
                    max_drop_k, max_drop = max(drops, key=lambda x: x[1])
                    
                    for k in k_values:
                        if k not in scores:
                            scores[k] = 0
                        
                        if k == max_drop_k:
                            scores[k] += weights['rel_change']
                            
                        elif k == max_drop_k - 1 or k == max_drop_k + 1:
                            scores[k] += weights['rel_change'] * 0.5
        
        elif metric_name == 'entropy' and 'entropy' in metrics[k_values[0]]:
            entropy_values = {k: metrics[k]['entropy'] for k in k_values if 'entropy' in metrics[k]}
            
            if entropy_values:
                max_entropy = max(entropy_values.values())
                min_entropy = min(entropy_values.values())
                range_entropy = max_entropy - min_entropy
                
                for k in entropy_values:
                    if k not in scores:
                        scores[k] = 0
                    if range_entropy > 0:
                        norm_score = 1 - (entropy_values[k] - min_entropy) / range_entropy
                    else:
                        norm_score = 1 if entropy_values[k] == min_entropy else 0
                    scores[k] += weights['entropy'] * norm_score
    
    if scores:
        optimal_k = max(scores.items(), key=lambda x: x[1])[0]
        return optimal_k
    else:
        return 5

def visualize_subtypes_umap(integrated_features, sample_subtypes):
    os.makedirs("results/figures/subtypes", exist_ok=True)
    
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(integrated_features)
    
    reducer = umap.UMAP(
        n_neighbors=30,
        min_dist=0.3,
        metric='correlation',
        random_state=42,
        n_components=2,
        low_memory=False
    )
    embedding = reducer.fit_transform(data_scaled)
    
    umap_df = pd.DataFrame({
        'UMAP1': embedding[:, 0],
        'UMAP2': embedding[:, 1],
        'sample': integrated_features.index
    })
    
    umap_df = umap_df.merge(sample_subtypes, on='sample')
    
    subtype_colors = {
        'Luminal A': 'royalblue',
        'Luminal B': 'lightblue',
        'Luminal B/HER2+': 'purple',
        'HER2-enriched': 'magenta',
        'Basal-like': 'red',
        'Normal-like': 'green'
    }
    
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(
        umap_df['UMAP1'], 
        umap_df['UMAP2'],
        c=umap_df['combined_confidence'], 
        cmap='viridis',
        alpha=0.8, 
        s=50, 
        vmin=0, 
        vmax=1
    )
    plt.colorbar(scatter, label='Classification Confidence')
    plt.title('UMAP Visualization Colored by Classification Confidence')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.savefig("results/figures/subtypes/umap_confidence.png", dpi=300)
    plt.close()
    
    umap_df.to_csv("results/figures/subtypes/umap_coordinates_with_confidence.tsv", sep='\t', index=False)
    
    return umap_df

def save_results(sample_subtypes, cluster_means, z_scores, subtype_map, clinical_data, subtype_confidence=None):
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("results/tables/subtypes", exist_ok=True)
    
    sample_subtypes.to_csv("data/processed/sample_subtypes.tsv", sep='\t', index=False)
    
    sample_subtypes[['sample', 'subtype', 'cluster_confidence', 'subtype_confidence', 
                   'combined_confidence', 'confidence_level']].to_csv(
        "data/processed/subtype_confidence_metrics.tsv", sep='\t', index=False)
    
    cluster_means.to_csv("data/processed/cluster_means.tsv", sep='\t')
    z_scores.to_csv("data/processed/cluster_z_scores.tsv", sep='\t')
    
    if subtype_confidence:
        subtype_mapping_df = pd.DataFrame(columns=['cluster', 'subtype', 'confidence', 'evidence'])
        
        for cluster, subtype in subtype_map.items():
            confidence_data = subtype_confidence[cluster]
            evidence_str = '; '.join(confidence_data['evidence'])
            
            subtype_mapping_df = subtype_mapping_df._append({
                'cluster': cluster,
                'subtype': subtype,
                'confidence': confidence_data['confidence'],
                'evidence': evidence_str,
                'similarity_scores': str(confidence_data['similarity_scores'])
            }, ignore_index=True)
        
        subtype_mapping_df.to_csv("data/processed/cluster_subtype_map_with_evidence.tsv", sep='\t', index=False)
    
    merged_data = clinical_data.merge(
        sample_subtypes, left_on='sample', right_on='sample', how='inner'
    )
    
    merged_data.to_csv("data/processed/clinical_with_subtypes.tsv", sep='\t', index=False)
    
    high_conf_data = merged_data[merged_data['confidence_level'] == 'High']
    high_conf_data.to_csv("data/processed/clinical_with_high_conf_subtypes.tsv", sep='\t', index=False)
    
    return merged_data

def analyze_survival(merged_data):
    os.makedirs("results/figures/survival", exist_ok=True)
    
    if 'OS' in merged_data.columns and 'OS.time' in merged_data.columns:
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
            kmf.plot(ci_show=True)
        
        plt.title('Overall Survival by Molecular Subtype (All Samples)')
        plt.xlabel('Time (days)')
        plt.ylabel('Survival Probability')
        plt.grid(alpha=0.3)
        plt.legend()
        plt.savefig("results/figures/survival/overall_survival_all_samples.png")
        plt.close()
        
        return True
    else:
        return False

def visualize_consensus_matrices(consensus_matrices, metrics):
    os.makedirs("results/figures/subtypes", exist_ok=True)

    for k, matrix in consensus_matrices.items():
        plt.figure(figsize=(10, 10))
        sns.heatmap(matrix, cmap='viridis', vmin=0, vmax=1)
        plt.title(f'Consensus Matrix for k={k}')
        plt.savefig(f"results/figures/subtypes/consensus_matrix_k{k}.png")
        plt.close()

if __name__ == "__main__":
    integrated_features = pd.read_csv("data/processed/integrated_features_imputed.tsv", sep='\t', index_col=0)

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(integrated_features)

    k_range = range(3, 6)
    consensus_matrices, connectivity_matrices, sample_occurrence = perform_consensus_clustering(data_scaled, n_iter=100, k_range=k_range)
    metrics, optimal_k = evaluate_clustering(consensus_matrices)
    visualize_consensus_matrices(consensus_matrices, metrics)

    cluster_assignments_df, silhouette_values, consensus_confidence = get_cluster_assignments(
        consensus_matrices[optimal_k], optimal_k, sample_names=integrated_features.index)
    
    cluster_means = characterize_clusters(integrated_features, cluster_assignments_df['cluster'], optimal_k)
    
    sample_subtypes, subtype_map, z_scores, subtype_confidence = assign_subtypes(
        cluster_means, integrated_features, cluster_assignments_df)
    
    umap_df = visualize_subtypes_umap(integrated_features, sample_subtypes)
    
    clinical_data = pd.read_csv("data/processed/matched_clinical.tsv", sep='\t')
    
    merged_data = save_results(sample_subtypes, cluster_means, z_scores, subtype_map, 
                              clinical_data, subtype_confidence)
    
    analyze_survival(merged_data)
