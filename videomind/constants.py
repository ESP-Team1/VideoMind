# Copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause License.

IGNORE_INDEX = -100

REG_TOKEN = '<|reg|>'

SEG_S_TOKEN = '<|seg_start|>'
SEG_E_TOKEN = '<|seg_end|>'

PLANNER_PROMPT = (
    'You are acting as the planner now. '
    'Given a question about the video, your task is to analyze the question and identify the best way to answer this question. '
    'You have access to the following tools:\n\n'
    'Grounder: Accepts a text query and localize the relevant video segment according to the query.\n'
    'Verifier: A tool supporting grounder by verifying the reliability of its outputs.\n'
    'Answerer: Answer a given question directly based on the whole video or a cropped video segment.\n\n'
    'Your response must be a list in JSON format. '
    'A valid plan for reasoning could be "grounder, verifier, answer", "grounder, verifier", or "answerer", depending on the given question. '
    'Please see an example for the format below.\n\n'
    '[{{"type": "grounder", "value": "<text query>"}}, {{"type": "verifier"}}, {{"type": "answerer"}}]\n\n'
    'Note that only the grounder can accept an argument called "value", which is the text query used for grounding. '
    "Now I give you the question: '{}'. "
    'Please think carefully and respond with your plan in JSON directly.')

GROUNDER_PROMPT = (
    'You are acting as the grounder now. '
    'Given a video and a text query, your goal is to temporally localize the video moment described by the query. '
    'If the query is directly describing a moment, simply localize it according to its content. '
    "Otherwise, if the moment is described as 'before/after a pivotal event', you need to determine the actual event it refers to. "
    "However, please return [0, 0] if you cannot find the actual event. "
    'The localized moment should only cover the target event. '
    "Now I give you the query: '{}'. "
    'Please think carefully and provide your response.')

VERIFIER_PROMPT = (
    'You are acting as the verifier now. '
    'You will be presented a text query describing a moment that potentialy happens in the given video. '
    f'Your task is to identify whether the video segment between {SEG_S_TOKEN} and {SEG_E_TOKEN} perfectly covers the moment. '
    f'If the described moment can be seen in the video, please focus on verifying whether the moment starts at {SEG_S_TOKEN} and ends at {SEG_E_TOKEN}. '
    "Respond with 'Yes' if you think the moment boundaries are correct, otherwise 'No'. "
    "If the described moment cannot be seen in the video, respond with 'No' directly. "
    "Now I give you the query: '{}'. "
    "Please think carefully and respond with 'Yes' or 'No' directly.")
