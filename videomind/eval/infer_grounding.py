import json
import os
import sys

import argparse
import torch
from videomind.constants import GROUNDER_PROMPT, VERIFIER_PROMPT
from videomind.dataset.utils import process_vision_info
from videomind.model.builder import build_model
from videomind.utils.parser import parse_query, parse_span
from segmenter import VideoSegmenter

def infer_grounding(video_path, query, model_gnd_path='model_zoo/VideoMind-7B', model_ver_path='model_zoo/VideoMind-7B', num_threads=1, device='cuda', segmenter=None, args=None):
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

    # Prepare the input message for the grounder
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

    # Process the input
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

    # Grounder inference
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
        max_new_tokens=256
    )

    assert data.input_ids.size(0) == output_ids.size(0) == 1
    output_ids = output_ids[0, data.input_ids.size(1):]
    if output_ids[-1] == processor.tokenizer.eos_token_id:
        output_ids = output_ids[:-1]
    response = processor.decode(output_ids, clean_up_tokenization_spaces=False)

    # Parse grounder response
    dump = {'video_path': video_path, 'query': query, 'grounder_response': response}
    dump['grounder_success'] = len(model.reg) > 0

    if dump['grounder_success']:
        # Extract timestamps and confidences
        duration = 1.0  # Placeholder for video duration; replace with actual duration if available
        blob = model.reg[0].cpu().float()
        pred, conf = blob[:, :2] * duration, blob[:, -1].tolist()

        # Clamp timestamps
        pred = pred.clamp(min=0, max=duration)

        # Round timestamps to units
        unit = 0.001
        pred = torch.round(pred / unit).long() * unit

        # Sort timestamps
        inds = (pred[:, 1] - pred[:, 0] < 0).nonzero()[:, 0]
        pred[inds] = pred[inds].roll(1)

        # Convert timestamps to list
        pred = pred.tolist()
    else:
        print('WARNING: Failed to parse grounder response')

        if adapter_state['verifier']:
            pred = [[i * duration / 6, (i + 2) * duration / 6] for i in range(5)]
            conf = [0] * 5
        else:
            pred = [[0, duration]]
            conf = [0]

    print(pred[0], duration)
    dump['pred'] = pred
    dump['conf'] = conf

    # Verifier logic
    if adapter_state['verifier'] and len(pred) > 1:
        print('=============== verifier ===============')

        probs = []
        for cand in pred[:5]:
            s0, e0 = parse_span(cand, duration, 2)
            offset = (e0 - s0) / 2
            s1, e1 = parse_span([s0 - offset, e0 + offset], duration)

            # Prepare verifier input
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

            # Insert segment start/end tokens
            video_grid_thw = data['video_grid_thw'][0]
            num_frames, window = int(video_grid_thw[0]), int(video_grid_thw[1] * video_grid_thw[2] / 4)
            assert num_frames * window * 4 == data['pixel_values_videos'].size(0)

            pos_s, pos_e = round(s0 * num_frames), round(e0 * num_frames)
            pos_s, pos_e = min(max(0, pos_s), num_frames), min(max(0, pos_e), num_frames)
            assert pos_s <= pos_e, (num_frames, s0, e0)

            base_idx = torch.nonzero(data['input_ids'][0] == model.config.vision_start_token_id).item()
            pos_s, pos_e = pos_s * window + base_idx + 1, pos_e * window + base_idx + 2

            input_ids = data['input_ids'][0].tolist()
            input_ids.insert(pos_s, model.config.seg_s_token_id)
            input_ids.insert(pos_e, model.config.seg_e_token_id)
            data['input_ids'] = torch.LongTensor([input_ids])
            data['attention_mask'] = torch.ones_like(data['input_ids'])

            data = data.to(device)

            model.base_model.disable_adapter_layers()
            model.base_model.enable_adapter_layers()
            model.set_adapter('verifier')

            with torch.inference_mode():
                logits = model(**data).logits[0, -1].softmax(dim=-1)

            # Calculate score
            score = (logits[9454] - logits[2753]).sigmoid().item()
            probs.append(score)

        # Rank predictions by verifier scores
        ranks = torch.Tensor(probs).argsort(descending=True).tolist()
        print(probs)
        print(ranks)

        pred = [pred[idx] for idx in ranks]
        conf = [conf[idx] for idx in ranks]
        dump['probs'] = probs
        dump['ranks'] = ranks
        dump['pred_ori'] = dump['pred']
        dump['conf_ori'] = dump['conf']
        dump['pred'] = pred
        dump['conf'] = conf

    return dump
    
def main():
    """
    Main function to test the infer_grounding function.
    """
    video_path = "videomind/eval/dataset/sj81PWrerDk.mp4" # Replace with your video path
    query = "The first two people doing the action"

    segmenter = VideoSegmenter()

    results = infer_grounding(video_path, query, segmenter=segmenter)

    print("Results:")
    print(json.dumps(results, indent=4))


if __name__ == "__main__":
    main()