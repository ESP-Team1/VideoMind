import json
import os
import torch
from videomind.constants import GROUNDER_PROMPT, VERIFIER_PROMPT
from videomind.dataset.utils import process_vision_info
from videomind.model.builder import build_model
from videomind.utils.parser import parse_query, parse_span
segmenter_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tools'))
sys.path.append(segmenter_dir)

from segmenter import segmenter

def infer_grounding(video_path, query, model_gnd_path='model_zoo/VideoMind-7B', model_ver_path='model_zoo/VideoMind-7B', num_threads=1, device='cuda', segmenter=None):
    """
    Runs the grounding and verification process for a given video and query.

    Args:
        video_path (str): Path to the video file.
        query (str): Query to evaluate.
        model_gnd_path (str): Path to the grounding model.
        model_ver_path (str): Path to the verifier model.
        num_threads (int): Number of threads for processing.
        device (str): Device to run the model on ('cuda' or 'cpu').

    Returns:
        dict: Results containing grounded timestamps, verifier probabilities, and the video path.
    """
    # Initialize the grounder model
    print('Initializing role *grounder*')
    model, processor = build_model(model_gnd_path, device=device)
    device = next(model.parameters()).device

    # Load verifier adapter if specified
    adapter_state = {'verifier': False}
    if model_ver_path is not None:
        adapter_path = os.path.join(model_ver_path, 'verifier')
        if os.path.isdir(adapter_path):
            print('Initializing role *verifier*')
            model.load_adapter(adapter_path, adapter_name='verifier')
            adapter_state['verifier'] = True

    print(f"Video: {video_path}")
    print(f"Query: {query}")

    # Grounder evaluation
    print('=============== grounder ===============')
    query = parse_query(query)

    messages = [{
        'role': 'user',
        'content': [{
            'type': 'video',
            'video': video_path,
            'num_threads': num_threads,
            'min_pixels': 36 * 28 * 28,
            'max_pixels': 64 * 28 * 28,
            'max_frames': 150,
            'fps': 2.0
        }, {
            'type': 'text',
            'text': GROUNDER_PROMPT.format(query)
        }]
    }]

    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    print(text)
    images, videos = process_vision_info(messages)
    data = processor(text=[text], images=images, videos=videos, return_tensors='pt')
    data = data.to(device)

    # Ensure proper batch dimensions and shapes
    if isinstance(data.input_ids, torch.Tensor):
        if data.input_ids.dim() == 1:
            data.input_ids = data.input_ids.unsqueeze(0)
        if data.attention_mask.dim() == 1:
            data.attention_mask = data.attention_mask.unsqueeze(0)
        if data.attention_mask.size(1) != data.input_ids.size(1):
            data.attention_mask = torch.ones_like(data.input_ids)

    model.base_model.disable_adapter_layers()
    model.base_model.enable_adapter_layers()
    model.set_adapter('grounder')

    output_ids = model.generate(
        **data,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        repetition_penalty=None,
        max_new_tokens=256)

    assert data.input_ids.size(0) == output_ids.size(0) == 1
    output_ids = output_ids[0, data.input_ids.size(1):]
    if output_ids[-1] == processor.tokenizer.eos_token_id:
        output_ids = output_ids[:-1]
    grounder_response = processor.decode(output_ids, clean_up_tokenization_spaces=False)
    print("Grounder response:", grounder_response)

    # Verifier evaluation
    probs = []
    if adapter_state['verifier']:
        print('=============== verifier ===============')

        # Example: Using the first prediction from the grounder
        pred = [[0, 10]]  # Replace with actual prediction from the grounder if available
        for cand in pred:
            s0, e0 = parse_span(cand, 10, 2)  # Replace 10 with actual video duration
            offset = (e0 - s0) / 2
            s1, e1 = parse_span([s0 - offset, e0 + offset], 10)

            messages = [{
                'role': 'user',
                'content': [{
                    'type': 'video',
                    'video': video_path,
                    'num_threads': num_threads,
                    'video_start': s1,
                    'video_end': e1,
                    'min_pixels': 36 * 28 * 28,
                    'max_pixels': 64 * 28 * 28,
                    'max_frames': 64,
                    'fps': 3.0
                }, {
                    'type': 'text',
                    'text': VERIFIER_PROMPT.format(query)
                }]
            }]

            text = processor.apply_chat_template(messages, add_generation_prompt=True)
            print(text)
            images, videos = process_vision_info(messages)
            data = processor(text=[text], images=images, videos=videos, return_tensors='pt')
            data = data.to(device)

            model.base_model.disable_adapter_layers()
            model.base_model.enable_adapter_layers()
            model.set_adapter('verifier')

            with torch.inference_mode():
                logits = model(**data).logits[0, -1].softmax(dim=-1)

            # NOTE: magic numbers here
            # In Qwen2-VL vocab: 9454 -> Yes, 2753 -> No
            score = (logits[9454] - logits[2753]).sigmoid().item()
            probs.append(score)

        print("Verifier probabilities:", probs)

    grounded_timestamps = segmenter.parse_timestamps(grounder_response)
    cropped_video_path = segmenter.extract_segments_with_blackout(video_path, grounded_timestamps[0], grounded_timestamps[1])
    results = {
        'cropped_video_path': cropped_video_path,
        'query': query,
        'grounder_response': grounder_response,
        'grounded_timestamps': grounded_timestamps,
        'verifier_probs': probs if adapter_state['verifier'] else None
    }

    return results

def main():
    """
    Main function to test the infer_grounding function.
    """
    video_path = "dataset/sj81PWrerDk.mp4"
    query = "The first two people doing the action"

    segmenter = segmenter.VideoSegmenter()

    results = infer_grounding(video_path, query, segmenter=segmenter)

    print("Results:")
    print(json.dumps(results, indent=4))


if __name__ == "__main__":
    main()