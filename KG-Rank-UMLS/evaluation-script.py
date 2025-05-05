import pandas as pd
import numpy as np
import os
import argparse
import torch
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import evaluate
import logging
import sys
import time
import warnings
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def compute_rouge(predictions, references):
    """Compute ROUGE-L scores"""
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = []
    
    for pred, ref in zip(predictions, references):
        if not isinstance(pred, str) or not isinstance(ref, str):
            continue
        
        try:
            score = scorer.score(ref, pred)
            scores.append(score['rougeL'].fmeasure)
        except Exception as e:
            logger.warning(f"Error computing ROUGE for a sample: {e}")
            continue
    
    if not scores:
        return 0.0
        
    return np.mean(scores) * 100  # Return as a percentage

def compute_bertscore(predictions, references):
    """Compute BERTScore"""
    # Filter out non-string inputs
    valid_pairs = [(p, r) for p, r in zip(predictions, references) 
                  if isinstance(p, str) and isinstance(r, str) and p.strip() and r.strip()]
    
    if not valid_pairs:
        return 0.0
    
    valid_preds, valid_refs = zip(*valid_pairs)
    
    try:
        P, R, F1 = bert_score(valid_preds, valid_refs, lang='en', rescale_with_baseline=True, batch_size=8)
        return F1.mean().item() * 100  # Return as a percentage
    except Exception as e:
        logger.error(f"Error computing BERTScore: {e}")
        return 0.0

def compute_moverscore(predictions, references):
    """Compute MoverScore"""
    try:
        from moverscore import get_idf_dict, word_mover_score
        
        # Filter out non-string inputs
        valid_pairs = [(p, r) for p, r in zip(predictions, references) 
                      if isinstance(p, str) and isinstance(r, str) and p.strip() and r.strip()]
        
        if not valid_pairs:
            return 0.0
        
        valid_preds, valid_refs = zip(*valid_pairs)
        
        # Process in smaller batches to avoid memory issues
        batch_size = 16
        scores = []
        
        for i in range(0, len(valid_preds), batch_size):
            batch_preds = valid_preds[i:i+batch_size]
            batch_refs = valid_refs[i:i+batch_size]
            
            # Prepare IDF dictionary for MoverScore
            idf_dict_hyp = get_idf_dict(batch_preds)
            idf_dict_ref = get_idf_dict(batch_refs)
            
            batch_scores = word_mover_score(batch_refs, batch_preds, idf_dict_ref, idf_dict_hyp, 
                                           stop_words=[], n_gram=1, remove_subwords=True)
            scores.extend(batch_scores)
        
        return np.mean(scores) * 100  # Return as a percentage
    except Exception as e:
        logger.warning(f"Error computing MoverScore: {e}")
        return 0.0

def compute_bleurt(predictions, references):
    """Compute BLEURT score"""
    # Filter out non-string inputs
    valid_pairs = [(p, r) for p, r in zip(predictions, references) 
                  if isinstance(p, str) and isinstance(r, str) and p.strip() and r.strip()]
    
    if not valid_pairs:
        return 0.0
    
    valid_preds, valid_refs = zip(*valid_pairs)
    
    try:
        bleurt = evaluate.load("bleurt")
        
        # Process in batches to avoid memory issues
        batch_size = 32
        all_scores = []
        
        for i in range(0, len(valid_preds), batch_size):
            batch_preds = valid_preds[i:i+batch_size]
            batch_refs = valid_refs[i:i+batch_size]
            
            results = bleurt.compute(predictions=batch_preds, references=batch_refs)
            all_scores.extend(results['scores'])
        
        return np.mean(all_scores) * 100  # Return as a percentage
    except Exception as e:
        logger.error(f"Error computing BLEURT: {e}")
        return 0.0

def run_evaluation(results_file):
    """Run evaluation on all metrics"""
    logger.info(f"Loading results from: {results_file}")
    
    # Load results
    results = pd.read_csv(results_file)
    
    # Check column names
    if 'Generated_Answer' not in results.columns or 'Reference_Answer' not in results.columns:
        logger.error("CSV file must contain 'Generated_Answer' and 'Reference_Answer' columns.")
        available_cols = ', '.join(results.columns)
        logger.error(f"Available columns: {available_cols}")
        return {}
    
    # Get predictions and references
    predictions = results['Generated_Answer'].tolist()
    references = results['Reference_Answer'].tolist()
    
    # Remove any empty or NaN values
    valid_pairs = [(p, r) for p, r in zip(predictions, references) 
                  if isinstance(p, str) and isinstance(r, str) and p.strip() and r.strip()]
    
    if not valid_pairs:
        logger.error("Error: No valid prediction-reference pairs found.")
        return {}
    
    logger.info(f"Found {len(valid_pairs)} valid prediction-reference pairs for evaluation.")
    valid_preds, valid_refs = zip(*valid_pairs)
    
    evaluation_results = {}
    
    # Compute ROUGE-L
    logger.info("Computing ROUGE-L...")
    start_time = time.time()
    rouge_l = compute_rouge(valid_preds, valid_refs)
    evaluation_results['ROUGE-L'] = rouge_l
    logger.info(f"ROUGE-L: {rouge_l:.2f} (computed in {time.time() - start_time:.2f}s)")
    
    # Compute BERTScore
    logger.info("Computing BERTScore...")
    start_time = time.time()
    bertscore_f1 = compute_bertscore(valid_preds, valid_refs)
    evaluation_results['BERTScore'] = bertscore_f1
    logger.info(f"BERTScore: {bertscore_f1:.2f} (computed in {time.time() - start_time:.2f}s)")
    
    # Compute MoverScore
    logger.info("Computing MoverScore...")
    start_time = time.time()
    try:
        moverscore_res = compute_moverscore(valid_preds, valid_refs)
        evaluation_results['MoverScore'] = moverscore_res
        logger.info(f"MoverScore: {moverscore_res:.2f} (computed in {time.time() - start_time:.2f}s)")
    except Exception as e:
        logger.warning(f"Failed to compute MoverScore: {e}")
        evaluation_results['MoverScore'] = 0.0
    
    # Compute BLEURT
    logger.info("Computing BLEURT...")
    start_time = time.time()
    bleurt_score = compute_bleurt(valid_preds, valid_refs)
    evaluation_results['BLEURT'] = bleurt_score
    logger.info(f"BLEURT: {bleurt_score:.2f} (computed in {time.time() - start_time:.2f}s)")
    
    # Print summary
    logger.info("\nEvaluation Summary:")
    for metric, score in evaluation_results.items():
        logger.info(f"{metric}: {score:.2f}")
    
    return evaluation_results

def compare_with_paper_results(computed_results, dataset_name):
    """Compare computed results with original paper results"""
    # Paper results from Table 2
    paper_results = {
        'MedicationQA': {
            'GPT-4 ZS': {'ROUGE-L': 14.41, 'BERTScore': 82.55, 'MoverScore': 52.62, 'BLEURT': 37.41},
            'GPT-4 Sim': {'ROUGE-L': 16.05, 'BERTScore': 83.56, 'MoverScore': 53.23, 'BLEURT': 37.60},
            'GPT-4 AE': {'ROUGE-L': 16.13, 'BERTScore': 83.46, 'MoverScore': 53.23, 'BLEURT': 37.87},
            'GPT-4 MMR': {'ROUGE-L': 15.89, 'BERTScore': 83.48, 'MoverScore': 53.22, 'BLEURT': 37.73},
            'GPT-4 RR': {'ROUGE-L': 16.19, 'BERTScore': 83.59, 'MoverScore': 53.30, 'BLEURT': 37.91}
        },
        'LiveQA': {
            'GPT-4 ZS': {'ROUGE-L': 18.89, 'BERTScore': 82.50, 'MoverScore': 54.02, 'BLEURT': 39.84},
            'GPT-4 Sim': {'ROUGE-L': 19.35, 'BERTScore': 83.01, 'MoverScore': 54.08, 'BLEURT': 40.47},
            'GPT-4 AE': {'ROUGE-L': 19.24, 'BERTScore': 82.95, 'MoverScore': 54.04, 'BLEURT': 40.15},
            'GPT-4 MMR': {'ROUGE-L': 19.32, 'BERTScore': 82.91, 'MoverScore': 54.03, 'BLEURT': 40.55},
            'GPT-4 RR': {'ROUGE-L': 19.44, 'BERTScore': 82.94, 'MoverScore': 54.11, 'BLEURT': 40.50}
        },
        'ExpertQA-Bio': {
            'GPT-4 ZS': {'ROUGE-L': 23.00, 'BERTScore': 84.50, 'MoverScore': 56.15, 'BLEURT': 44.53},
            'GPT-4 Sim': {'ROUGE-L': 25.90, 'BERTScore': 85.72, 'MoverScore': 56.73, 'BLEURT': 45.10},
            'GPT-4 AE': {'ROUGE-L': 26.78, 'BERTScore': 85.77, 'MoverScore': 56.79, 'BLEURT': 45.18},
            'GPT-4 MMR': {'ROUGE-L': 26.54, 'BERTScore': 85.76, 'MoverScore': 56.77, 'BLEURT': 44.93},
            'GPT-4 RR': {'ROUGE-L': 27.20, 'BERTScore': 85.83, 'MoverScore': 57.11, 'BLEURT': 45.91}
        },
        'ExpertQA-Med': {
            'GPT-4 ZS': {'ROUGE-L': 25.45, 'BERTScore': 85.11, 'MoverScore': 56.50, 'BLEURT': 45.98},
            'GPT-4 Sim': {'ROUGE-L': 27.61, 'BERTScore': 86.10, 'MoverScore': 57.13, 'BLEURT': 46.47},
            'GPT-4 AE': {'ROUGE-L': 27.98, 'BERTScore': 86.12, 'MoverScore': 57.25, 'BLEURT': 46.80},
            'GPT-4 MMR': {'ROUGE-L': 27.78, 'BERTScore': 86.22, 'MoverScore': 57.28, 'BLEURT': 46.84},
            'GPT-4 RR': {'ROUGE-L': 28.08, 'BERTScore': 86.30, 'MoverScore': 57.32, 'BLEURT': 47.00}
        },
        'MedQA': {  # Alias for MedicationQA
            'GPT-4 ZS': {'ROUGE-L': 14.41, 'BERTScore': 82.55, 'MoverScore': 52.62, 'BLEURT': 37.41},
            'GPT-4 Sim': {'ROUGE-L': 16.05, 'BERTScore': 83.56, 'MoverScore': 53.23, 'BLEURT': 37.60},
            'GPT-4 AE': {'ROUGE-L': 16.13, 'BERTScore': 83.46, 'MoverScore': 53.23, 'BLEURT': 37.87},
            'GPT-4 MMR': {'ROUGE-L': 15.89, 'BERTScore': 83.48, 'MoverScore': 53.22, 'BLEURT': 37.73},
            'GPT-4 RR': {'ROUGE-L': 16.19, 'BERTScore': 83.59, 'MoverScore': 53.30, 'BLEURT': 37.91}
        }
    }
    
    if dataset_name in paper_results:
        logger.info(f"\nComparison with original KG-Rank paper results for {dataset_name}:")
        
        # Create table header
        logger.info(f"{'Metric':<12} {'Your Results':<15} {'GPT-4 ZS':<15} {'GPT-4 Sim':<15} {'GPT-4 RR':<15}")
        logger.info("-" * 72)
        
        for metric, value in computed_results.items():
            paper_zs = paper_results[dataset_name]['GPT-4 ZS'][metric]
            paper_sim = paper_results[dataset_name]['GPT-4 Sim'][metric]
            paper_rr = paper_results[dataset_name]['GPT-4 RR'][metric]
            
            logger.info(f"{metric:<12} {value:<15.2f} {paper_zs:<15.2f} {paper_sim:<15.2f} {paper_rr:<15.2f}")
        
        # Calculate and show differences
        logger.info("\nDifferences (Yours - Paper):")
        logger.info(f"{'Metric':<12} {'vs ZS':<15} {'vs Sim':<15} {'vs RR':<15}")
        logger.info("-" * 60)
        
        for metric, value in computed_results.items():
            paper_zs = paper_results[dataset_name]['GPT-4 ZS'][metric]
            paper_sim = paper_results[dataset_name]['GPT-4 Sim'][metric]
            paper_rr = paper_results[dataset_name]['GPT-4 RR'][metric]
            
            diff_zs = value - paper_zs
            diff_sim = value - paper_sim
            diff_rr = value - paper_rr
            
            # Calculate percentage differences
            pct_zs = diff_zs/paper_zs*100 if paper_zs != 0 else float('inf')
            pct_sim = diff_sim/paper_sim*100 if paper_sim != 0 else float('inf')
            pct_rr = diff_rr/paper_rr*100 if paper_rr != 0 else float('inf')
            
            logger.info(f"{metric:<12} {diff_zs:+.2f} ({pct_zs:+.1f}%) {diff_sim:+.2f} ({pct_sim:+.1f}%) {diff_rr:+.2f} ({pct_rr:+.1f}%)")
    else:
        logger.warning(f"No comparison available for {dataset_name} in the paper results.")

def save_results(results, results_file, output_dir='evaluation_results'):
    """Save evaluation results to file"""
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.basename(results_file).split('.')[0]
    output_file = f"{output_dir}/{base_name}_evaluation.csv"
    
    # Convert results to DataFrame
    results_df = pd.DataFrame([results])
    
    # Save to CSV
    results_df.to_csv(output_file, index=False)
    logger.info(f"Evaluation results saved to: {output_file}")
    
    # Also save results in a format suitable for paper comparison
    comparison_file = f"{output_dir}/{base_name}_comparison.csv"
    
    # Create a dataframe with the same format as the paper table
    comparison_df = pd.DataFrame({
        'Dataset': [os.path.splitext(os.path.basename(results_file))[0]],
        'Method': ['Ours'],
        'ROUGE-L': [results['ROUGE-L']],
        'BERTScore': [results['BERTScore']],
        'MoverScore': [results['MoverScore']],
        'BLEURT': [results['BLEURT']]
    })
    
    comparison_df.to_csv(comparison_file, index=False)
    logger.info(f"Comparison results saved to: {comparison_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate medical QA results and compare with KG-Rank paper.')
    parser.add_argument('--results', type=str, required=True, 
                        help='Path to the results CSV file with Generated_Answer and Reference_Answer columns')
    parser.add_argument('--dataset', type=str, required=True, 
                        choices=['MedQA', 'MedicationQA', 'LiveQA', 'ExpertQA-Bio', 'ExpertQA-Med'], 
                        help='Dataset name for comparison with paper results')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                        help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    logger.info(f"Starting evaluation for {args.dataset} dataset")
    
    # Run evaluation
    results = run_evaluation(args.results)
    
    if results:
        # Compare with paper results
        compare_with_paper_results(results, args.dataset)
        
        # Save results
        save_results(results, args.results, args.output_dir)
        
        logger.info("Evaluation completed successfully.")
