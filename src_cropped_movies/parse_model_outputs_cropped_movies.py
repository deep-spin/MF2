import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, Optional
from loguru import logger

def parse_args():
    parser = argparse.ArgumentParser(description="Process model results from JSON files")
    parser.add_argument("--output_dir_cropped", required=True,
                       help="Ouput directory containing results on cropped movies.")
    parser.add_argument("--output_dir_full", default=None,
                       help="Output directory containing results on full movies.")
    parser.add_argument("--strategy", required=False, default="strict", choices=["strict", "first-occurrence", "last-occurrence", "strict-w-fallback-first-occurrence"],
                        help="Parsing strategy")
    return parser.parse_args()


def read_results(output_dir):    
    output_dir = Path(output_dir)  # Convert string to Path object
    
    if not output_dir.exists():
        logger.error(f"Directory not found: {output_dir}")
        return None
    
    all_results = {}

    # Read all JSON files in the directory
    for json_file in output_dir.glob("*-results.json"):
        try:
            movie_id = int(json_file.stem.split('-')[0])  # Extract movie ID from filename
            with open(json_file, 'r') as f:
                data = json.load(f)
                all_results[movie_id] = data
        except Exception as e:
            logger.error(f"Error processing file {json_file}: {str(e)}")
            continue
    return all_results


def parse_model_output(output: str, strategy: str = "strict") -> Optional[str]:
    """
    Parse model output based on specified strategy.
    
    Args:
        output (str): The model's output string
        strategy (str): Either "strict" or "free-form"
    
    Returns:
        Optional[str]: The parsed answer ("TRUE" or "FALSE") or None if no match found
    """
    if not output:
        return None
        
    if strategy == "strict":
        # Look for answer between <answer> and </answer> tags
        # Case insensitive and handles potential variations in tag format
        pattern = r'<answer>(.*?)</answer>'
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            answer = match.group(1).strip().upper()
            return answer if answer in ["TRUE", "FALSE"] else None
            
    elif strategy == "first-occurrence":
        # Find first occurrence of true/false in the text
        match = re.search(r'\b(true|false)\b', output.lower())
        if match:
            return match.group(1).upper()
    elif strategy == "last-occurrence":
        # Find last occurrence of true/false in the text
        matches = list(re.finditer(r'\b(true|false)\b', output.lower()))
        if matches:
            return matches[-1].group(1).upper()
    elif strategy == "strict-w-fallback-first-occurrence":
        # Try strict first
        pattern = r'<answer>(.*?)</answer>'
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            answer = match.group(1).strip().upper()
            if answer in ["TRUE", "FALSE"]:
                return answer
        # Fallback to first-occurrence
        match = re.search(r'\b(true|false)\b', output.lower())
        if match:
            return match.group(1).upper()
    return None


def process_results(results: Dict, strategy: str = "strict") -> Dict:
    """
    Process all results using specified parsing strategy.
    
    Args:
        results (Dict): Dictionary of results by movie_id
        strategy (str): Parsing strategy ("strict" or "first-occurrence" or "last-occurrence")
    
    Returns:
        Dict: Processed results with parsed answers
    """
    processed_results = {}
    
    for movie_id, movie_data in results.items():
        movie_processed = {}
        for claim_pair_id, claims in movie_data.items():
            # Handle both formats: "true_claim"/"false_claim" and "Claim_1"/"Claim_2"
            claim1_text = claims.get("true_claim") or claims.get("Claim_1")
            claim2_text = claims.get("false_claim") or claims.get("Claim_2")
            
            claim1_answer = parse_model_output(claim1_text, strategy)
            claim2_answer = parse_model_output(claim2_text, strategy)
            
            movie_processed[claim_pair_id] = {
                "Claim_1": claim1_answer,
                "Claim_2": claim2_answer
            }
        processed_results[movie_id] = movie_processed
    return processed_results


def calculate_metrics(processed_results: Dict):
    total_outputs = 0
    none_count = 0
    true_positives = 0
    true_negatives = 0
    correct_pairs = 0
    total_pairs = 0
    
    # Store per-movie results
    movie_results = {}
    
    for movie_id, movie_data in processed_results.items():
        movie_tp = 0
        movie_tn = 0
        movie_correct_pairs = 0
        movie_total_pairs = 0
        movie_none_count = 0
        movie_outputs = 0
        
        for claim_pair_id, claims in movie_data.items():
            if claims["Claim_1"] is None:
                none_count += 1
                movie_none_count += 1
            if claims["Claim_2"] is None:
                none_count += 1
                movie_none_count += 1
            total_outputs += 2
            movie_outputs += 2
            
                
            if claims["Claim_1"] == "TRUE":
                true_positives += 1
                movie_tp += 1
                
            if claims["Claim_2"] == "FALSE":
                true_negatives += 1
                movie_tn += 1
                
            if claims["Claim_1"] == "TRUE" and claims["Claim_2"] == "FALSE":
                correct_pairs += 1
                movie_correct_pairs += 1

            total_pairs += 1
            movie_total_pairs += 1
        
        # Store movie-level results
        if movie_total_pairs > 0:
            movie_results[movie_id] = {
                "none_rate": (movie_none_count / movie_outputs) * 100 if movie_outputs > 0 else 0,
                "true_positive_rate": (movie_tp / movie_total_pairs) * 100 if movie_total_pairs > 0 else 0,
                "true_negative_rate": (movie_tn / movie_total_pairs) * 100 if movie_total_pairs > 0 else 0,
                "pairwise_accuracy": (movie_correct_pairs / movie_total_pairs) * 100 if movie_total_pairs > 0 else 0,
                "total_pairs": movie_total_pairs,
                "total_outputs": movie_outputs
            }
    
    # Calculate overall metrics
    none_percentage = (none_count / total_outputs) * 100 if total_outputs > 0 else 0
    tp_rate = (true_positives / total_pairs) * 100 if total_pairs > 0 else 0
    tn_rate = (true_negatives / total_pairs) * 100 if total_pairs > 0 else 0
    pairwise_accuracy = (correct_pairs / total_pairs) * 100 if total_pairs > 0 else 0
    accuracy = (true_positives + true_negatives) /  total_outputs * 100 if total_outputs > 0 else 0

    return {
        "overall_metrics": {
            "none_rate": none_percentage,
            "true_positive_rate": tp_rate,
            "true_negative_rate": tn_rate,
            "accuracy": accuracy,
            "pairwise_accuracy": pairwise_accuracy,
            "total_pairs": total_pairs,
            "total_outputs": total_outputs,
        },
        "per_movie_metrics": movie_results
    }



def write_results(metrics, processed_results, output_dir, strategy):
    # Create metrics directory if it doesn't exist
    # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_file = os.path.join(output_dir, f"{strategy}_metrics.json")
    processed_file = os.path.join(output_dir, f"{strategy}_parsed_results.json")
    
    
    # Write metrics to file
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Write processed results to file
    with open(processed_file, 'w') as f:
        json.dump(processed_results, f, indent=4)


def filter_results_on_subset(
    reference_results: Dict,
    target_results: Dict
) -> Dict:
    """
    Filter target_results so that only movie_ids and claim_pair_ids
    present in reference_results are kept.
    """
    filtered_results = {}

    for movie_id, ref_movie_data in reference_results.items():
        if movie_id not in target_results:
            continue

        target_movie_data = target_results[movie_id]
        filtered_movie_data = {}

        for claim_pair_id in ref_movie_data.keys():
            if claim_pair_id in target_movie_data:
                filtered_movie_data[claim_pair_id] = target_movie_data[claim_pair_id]

        if filtered_movie_data:
            filtered_results[movie_id] = filtered_movie_data

    return filtered_results
    

def main():
    args = parse_args()

    # ---------- CROPPED MOVIES RESULTS ----------
    logger.info(f"Reading results on cropped movies from: {args.output_dir_cropped}")
    results_cropped = read_results(args.output_dir_cropped)
    processed_cropped = process_results(results_cropped, args.strategy)
    metrics_cropped = calculate_metrics(processed_cropped)
    logger.info(f"[Results on Cropped Movies] Overall Metrics: {metrics_cropped['overall_metrics']}")

    # ---------- FULL MOVIES RESULTS (OPTIONAL) ----------
    if args.output_dir_full:
        logger.info(f"Reading results on full movies from: {args.output_dir_full}")
        results_full = read_results(args.output_dir_full)
        # Filter output_dir2 based on output_dir1
        subset_results_full = filter_results_on_subset(
            reference_results=results_cropped,
            target_results=results_full
        )

        processed_full = process_results(subset_results_full, args.strategy)
        metrics_full = calculate_metrics(processed_full)
        logger.info(f"[Results on Full Movies] Overall Metrics: {metrics_full['overall_metrics']}")

    # # Save metrics to a JSON file
    # write_results(metrics, processed_results, args.output_dir, args.strategy)

if __name__ == "__main__":
    main()