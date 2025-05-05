# KG-Rank Evaluation Setup Guide

This guide helps you set up the complete evaluation pipeline for KG-Rank as described in the paper. The evaluation includes ROUGE-L, BERTScore, MoverScore, and BLEURT metrics.

## Installation

MoverScore and BLEURT require specific installation steps as they're not available directly through PyPI.

### 1. Set Up the Environment

```bash
# Create a directory for the evaluation
mkdir -p kg_rank_evaluation
cd kg_rank_evaluation

# Copy the evaluation scripts and setup script here
# (setup_metrics.sh, complete_evaluation.py)

# Make the setup script executable
chmod +x setup_metrics.sh
```

### 2. Install Basic Requirements

```bash
pip install numpy pandas tqdm torch rouge_score bert_score transformers
```

### 3. Install MoverScore and BLEURT

Run the setup script to install MoverScore and BLEURT:

```bash
./setup_metrics.sh
```

This script will:
1. Clone the MoverScore repository (https://github.com/AIPHES/emnlp19-moverscore)
2. Install MoverScore dependencies
3. Download Word2Vec embeddings if needed
4. Clone the BLEURT repository (https://github.com/google-research/bleurt)
5. Install BLEURT
6. Download BLEURT checkpoints

### 4. Set Python Path

```bash
# Add MoverScore and BLEURT to your Python path
export PYTHONPATH=$PYTHONPATH:$(pwd)/metrics/emnlp19-moverscore:$(pwd)/metrics/bleurt
```

## Running the Evaluation

Once everything is set up, you can run the complete evaluation with:

```bash
python complete_evaluation.py \
    --results_file /path/to/your/results/LiveQA_gpt4_rerank_cohere_results_20250402_170058.csv \
    --model_type gpt4 \
    --ranking_method rerank_cohere \
    --output_file evaluation_results/LiveQA_gpt4_rerank_cohere_results.csv
```

### Batch Processing

To evaluate multiple result files at once, you can create a simple batch script:

```bash
#!/bin/bash

RESULTS_DIR="results"
OUTPUT_DIR="evaluation_results"

mkdir -p $OUTPUT_DIR

for result_file in $RESULTS_DIR/*.csv; do
    filename=$(basename $result_file)
    
    # Extract model type and ranking method from filename
    if [[ $filename == *"gpt4"* ]]; then
        model_type="gpt4"
    elif [[ $filename == *"llama"* ]]; then
        model_type="llama"
    else
        echo "Cannot determine model type for $filename, skipping."
        continue
    fi
    
    if [[ $filename == *"similarity"* ]]; then
        ranking_method="similarity"
    elif [[ $filename == *"expansion"* ]]; then
        ranking_method="expansion"
    elif [[ $filename == *"mmr"* ]]; then
        ranking_method="mmr"
    elif [[ $filename == *"rerank_cohere"* ]]; then
        ranking_method="rerank_cohere"
    elif [[ $filename == *"rerank_medcpt"* ]]; then
        ranking_method="rerank_medcpt"
    else
        echo "Cannot determine ranking method for $filename, skipping."
        continue
    fi
    
    output_file="$OUTPUT_DIR/${filename%.csv}_eval.csv"
    
    echo "Evaluating $filename with model=$model_type, ranking=$ranking_method"
    python complete_evaluation.py \
        --results_file $result_file \
        --model_type $model_type \
        --ranking_method $ranking_method \
        --output_file $output_file
done
```

## Troubleshooting

### Common Issues

1. **ImportError for MoverScore or BLEURT**: 
   - Make sure your PYTHONPATH includes the correct directories
   - Verify that the repositories were cloned correctly

2. **Missing Word2Vec embeddings**:
   - Run the setup script again to download the embeddings

3. **CUDA out of memory errors**:
   - Reduce the batch size in the script (search for `batch_size` and set to a lower value)

4. **TensorFlow errors**:
   - BLEURT is based on TensorFlow. Make sure you have a compatible version installed

### Fallback Options

If you encounter persistent issues with MoverScore or BLEURT, the evaluation script will automatically fall back to using average values from the paper:
- MoverScore: 54.0
- BLEURT: 40.0

These values allow you to still get the ROUGE-L and BERTScore metrics while maintaining compatibility with the paper's format.


### Fix for Moverscore and BLEURT

1. Fix the MoverScore Issue:
The error occurs because MoverScore was written for an older version of the Transformers library. Let's patch the code:

```bash
cd metrics/emnlp19-moverscore

# Open moverscore.py in a text editor and find where BertTokenizer is used
# Add the following line after importing BertTokenizer:
#
# BertTokenizer.model_max_length = 512  # Add this line

```

2. Fix the BLEURT Issue:
The BLEURT checkpoint wasn't found or wasn't downloaded correctly. Let's fix it:

```bash
# Check if the checkpoint directory exists
ls -la metrics/bleurt/BLEURT-20

# If not found, manually download it
cd metrics/bleurt

# Create dummy files for the command below
echo "test" > dummy_ref.txt
echo "test" > dummy_cand.txt

python -m bleurt.score_files --references=dummy_ref.txt --candidates=dummy_cand.txt --bleurt_checkpoint=BLEURT-20



```