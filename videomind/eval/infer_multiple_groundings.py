import subprocess
import json
import os
import argparse
import pandas as pd 
import sys

segmenter_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tools'))
sys.path.append(segmenter_dir)

from segmenter import segmenter


def run_infer_grounding(video_path, query, output_path):
    """
    Calls infer_grounding.py with the given video path and query.
    
    Args:
        video_path (str): Path to the input video.
        query (str): Query to evaluate.
        output_path (str): Path to save the results.

    Returns:
        dict: The results from infer_grounding.py, including grounded timestamps and new video path.
    """
    command = [
        "python", "infer_grounding.py",
        "--video_path", video_path,
        "--query", query,
        "--output_path", output_path
    ]
    subprocess.run(command, check=True)

    with open(output_path, "r") as f:
        results = json.load(f)
    return results


def process_queries(video_path, queries, temp_dir="temp_results", segmenter=None):
    """
    Processes a list of queries sequentially on a single video.

    Args:
        video_path (str): Path to the initial video.
        queries (list): List of queries to process.
        temp_dir (str): Directory to store temporary results.

    Returns:
        dict: A dictionary where keys are queries and values are grounded timestamps.
    """
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    results = {}
    current_video_path = video_path

    for i, query in enumerate(queries):
        print(f"Processing query {i + 1}/{len(queries)}: {query}")

        output_path = os.path.join(temp_dir, f"result_query_{i + 1}.json")

        grounding_result = run_infer_grounding(current_video_path, query, output_path, segmenter)

        grounded_timestamps = grounding_result.get("grounder_response", [])
        new_video_path = grounding_result.get("cropped_video_path", current_video_path)

        results[query] = grounded_timestamps

        current_video_path = new_video_path

    return results


def parse_args():
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Process multiple queries on a single video using infer_grounding.py.")
    parser.add_argument("--video_path", default="dataset/sj81PWrerDk.mp4", help="Path to the initial video.")
    parser.add_argument("--query_csv", default="dataset/queries.xlsx", help="Path to the CSV file containing queries.")
    parser.add_argument("--temp_dir", default="temp_results", help="Directory to store temporary results.")
    parser.add_argument("--output_path", default="results/final_results.json", help="Path to save the final results.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    queries_df = pd.read_csv(args.query_csv)
    if "query" not in queries_df.columns:
        raise ValueError("The CSV file must contain a 'query' column.")
    queries = queries_df["query"].tolist()

    segmenter = segmenter.VideoSegmenter()
    final_results = process_queries(args.video_path, queries, args.temp_dir, segmenter)

    with open(args.output_path, "w") as f:
        json.dump(final_results, f, indent=4)