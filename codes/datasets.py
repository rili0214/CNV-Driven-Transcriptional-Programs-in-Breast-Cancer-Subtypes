# data_download.py
import os
import requests
import gzip
import shutil

def create_directories():
    dirs = [
        "data/raw",
        "data/processed",
        "results/figures",
        "results/tables"
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    print("Directory structure created successfully.")

def download_xena_data(dataset_id, output_file):
    base_url = "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/"
    url = base_url + dataset_id
    print(f"Downloading {url}...")
    
    response = requests.get(url)
    if response.status_code == 200:
        with open(output_file, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded to {output_file}")
        return True
    else:
        print(f"ERROR: Failed to download {url}")
        return False

def decompress_gz(gz_file, output_file):
    with gzip.open(gz_file, 'rb') as f_in:
        with open(output_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    print(f"Decompressed {gz_file} to {output_file}")

def acquire_data():
    # 1. Download GISTIC2 thresholded CNV data
    cnv_dataset = "TCGA.BRCA.sampleMap%2FGistic2_CopyNumber_Gistic2_all_thresholded.by_genes.gz"
    cnv_output = "data/raw/brca_cnv_gistic2_thresholded.gz"
    cnv_decompressed = "data/raw/brca_cnv_gistic2_thresholded.tsv"
    
    # 2. Download RNAseq gene expression data
    expr_dataset = "TCGA.BRCA.sampleMap%2FHiSeqV2.gz"
    expr_output = "data/raw/brca_expression_rnaseq.gz"
    expr_decompressed = "data/raw/brca_expression_rnaseq.tsv"
    
    # 3. Download RPPA protein expression data
    rppa_dataset = "TCGA.BRCA.sampleMap%2FRPPA.gz"
    rppa_output = "data/raw/brca_rppa.gz"
    rppa_decompressed = "data/raw/brca_rppa.tsv"
    
    # 4. Download clinical data
    clinical_dataset = "survival%2FBRCA_survival.txt"
    clinical_output = "data/raw/brca_clinical.txt"
    
    # 5. Download PARADIGM data, will check if need this later
    paradigm_dataset = "merge_merged_reals%2FBRCA_merge_merged_reals.txt.gz"
    paradigm_output = "data/raw/brca_paradigm.gz"
    paradigm_decompressed = "data/raw/brca_paradigm.tsv"

    download_xena_data(cnv_dataset, cnv_output)
    download_xena_data(expr_dataset, expr_output)
    download_xena_data(rppa_dataset, rppa_output)
    download_xena_data(clinical_dataset, clinical_output)
    download_xena_data(paradigm_dataset, paradigm_output)

    decompress_gz(cnv_output, cnv_decompressed)
    decompress_gz(expr_output, expr_decompressed)
    decompress_gz(rppa_output, rppa_decompressed)
    decompress_gz(paradigm_output, paradigm_decompressed)
    
    print("All datasets downloaded and decompressed successfully.")
    
    return {
        'cnv': cnv_decompressed,
        'expression': expr_decompressed,
        'rppa': rppa_decompressed,
        'clinical': clinical_output,
        'paradigm': paradigm_decompressed
    }

if __name__ == "__main__":
    print("Part 1. Data Download")
    create_directories()
    file_paths = acquire_data()
    print("Data download and decompression completed successfully!")
    print("File paths for preprocessing:")
    for data_type, path in file_paths.items():
        print(f"- {data_type}: {path}")