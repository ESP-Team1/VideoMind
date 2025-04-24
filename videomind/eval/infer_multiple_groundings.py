import subprocess
import json
import os
import argparse
import pandas as pd 
import sys

from infer_grounding import infer_grounding
from segmenter import VideoSegmenter

def query_parser(query_csv):
    """
    Parses the query CSV file and returns a list of queries.

    Args:
        query_csv (str): Path to the CSV file containing queries.

    Returns:
        list: A list of queries.
    """
    df = pd.read_excel(query_csv)
    queries = df['query'].tolist()
    return queries

def process_queries(video_path, query_csv, temp_dir="temp_results", segmenter=None):
    """
    Processes a list of queries sequentially on a single video.

    Args:
        video_path (str): Path to the initial video.
        query_csv (str): Path to the CSV file containing queries.
        temp_dir (str): Directory to store temporary results.
        segmenter (VideoSegmenter): Segmenter instance for extracting video segments.

    Returns:
        dict: A dictionary where keys are queries and values are the extracted results.
    """
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    queries = query_parser(query_csv)

    results_dict = {}
    current_video_path = video_path

    for i, query in enumerate(queries):
        print(f"Processing query {i + 1}/{len(queries)}: {query}")

        # Call infer_grounding
        grounding_result = infer_grounding(current_video_path, query, segmenter=segmenter)

        # Save the first-ranked predicted duration
        first_ranked_pred = grounding_result.get("pred", [[0, 0]])[0]  # Default to [0, 0] if no predictions

        # Update the current video path if needed
        cropped_video_path = grounding_result.get("cropped_video_path", current_video_path)
        predicted_time_stamp = first_ranked_pred

        # Extract video segments using the segmenter
        extracted_results = segmenter.extract_segments_with_blackout(
            video_path, first_ranked_pred[0], first_ranked_pred[1]
        )
        current_video_path = extracted_results["leftover_video"].get("path")

        # Save the result to temp_dir
        output_path = os.path.join(temp_dir, f"result_query_{i + 1}.json")
        with open(output_path, 'w') as f:
            json.dump(extracted_results, f)

        # Add the extracted results to the dictionary with the query as the key
        results_dict[query] = extracted_results

    return results_dict

def parse_args():
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Process multiple queries on a single video using infer_grounding.py.")
    parser.add_argument("--video_path", default="videomind/eval/dataset/sj81PWrerDk.mp4", help="Path to the initial video.")
    parser.add_argument("--query_csv", default="videomind/eval/dataset/queries.xlsx", help="Path to the CSV file containing queries.")
    parser.add_argument("--temp_dir", default="temp_results", help="Directory to store temporary results.")
    parser.add_argument("--output_path", default="results/final_results.json", help="Path to save the final results.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    segmenter = VideoSegmenter()
    final_results = process_queries(args.video_path, args.query_csv, args.temp_dir, segmenter)

    print("Final results:")
    print(final_results)