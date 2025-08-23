# author = liuwei
# date=2025-04-01

import os
import json

coherence_levels = {
    "1": "low",
    "2": "medium",
    "3": "high",
    "low": "low",
    "medium": "medium",
    "high": "high"
}

label2idx = {
    "low": 0,
    "medium": 1,
    "high": 2
}

def data_to_alpaca_format(
    data_list,
    instruction_template,
    response_template
):
    new_data_list = []
    for idx, data in enumerate(data_list):
        text = data["text"]
        score = data["score"]

        instruction = instruction_template.render(text=text)
        response = response_template.render(level=coherence_levels[score])
        # response = response_template.render(level=score)

        sample = {
            'id': f'{idx}',
            'system_prompt': 'You are an AI assistant tasked with coherence assessment.',
            'instruction': instruction,
            'output': response
        }
        new_data_list.append(sample)

    return new_data_list

def data_to_alpaca_format_withrel(
    data_list,
    instruction_template,
    response_template,
    no_rel=False,
    no_enty=False,
    max_num=32
):
    new_data_list = []
    for idx, data in enumerate(data_list):
        text = data["text"]
        sents = data["sents"]
        score = data["score"]
        rels = data["rels"]
        spans = data["spans"]

        sentences = [(idx+1, sent) for idx, sent in enumerate(sents)]
        relations = []
        for rel, span in zip(rels, spans):
            if span[0] == span[1]:
                continue

            if no_rel:
                if rel == "entity":
                    relations.append((span[0]+1, rel, span[1]+1))
            elif no_enty:
                if rel != "entity":
                    relations.append((span[0] + 1, rel, span[1] + 1))
            else:
                relations.append((span[0] + 1, rel, span[1] + 1))
        relations = relations[:max_num]
        instruction = instruction_template.render(sentences=sentences, relations=relations)
        response = response_template.render(level=coherence_levels[score])

        sample = {
            'id': f'{idx}',
            'system_prompt': 'You are an AI assistant tasked with coherence assessment.',
            'instruction': instruction,
            'output': response
        }
        new_data_list.append(sample)

    return new_data_list

def data_to_alpaca_format_sent(
    data_list,
    instruction_template,
    response_template
):
    new_data_list = []
    for idx, data in enumerate(data_list):
        text = data["text"]
        sents = data["sents"]
        score = data["score"]
        rels = data["rels"]
        spans = data["spans"]

        sentences = [(idx+1, sent) for idx, sent in enumerate(sents)]
        instruction = instruction_template.render(sentences=sentences)
        response = response_template.render(level=coherence_levels[score])

        sample = {
            'id': f'{idx}',
            'system_prompt': 'You are an AI assistant tasked with coherence assessment.',
            'instruction': instruction,
            'output': response
        }
        new_data_list.append(sample)

    return new_data_list

def data_to_alpace_format_zero(
    data_list,
    instruction_template,
    response_template
):
    new_data_list = []
    for idx, data in enumerate(data_list):
        text = data["text"]
        score = data["score"]

        instruction = instruction_template.render(text=text)
        response = response_template.render(level=coherence_levels[score])

        sample = {
            'id': f'{idx}',
            'system_prompt': 'You are an AI assistant tasked with coherence assessment.',
            'instruction': instruction,
            'output': response
        }
        new_data_list.append(sample)

    return new_data_list


def data_to_alpace_format_zero_withrel(
    data_list,
    instruction_template,
    response_template,
    no_rel=False,
    no_enty=False,
    max_num=32
):
    new_data_list = []
    for idx, data in enumerate(data_list):
        text = data["text"]
        sents = data["sents"]
        score = data["score"]
        rels = data["rels"]
        spans = data["spans"]

        sentences = [(idx + 1, sent) for idx, sent in enumerate(sents)]
        relations = []
        for rel, span in zip(rels, spans):
            if span[0] == span[1]:
                continue

            if no_rel:
                if rel == "entity":
                    relations.append((span[0] + 1, rel, span[1] + 1))
            elif no_enty:
                if rel != "entity":
                    relations.append((span[0] + 1, rel, span[1] + 1))
            else:
                relations.append((span[0] + 1, rel, span[1] + 1))
        relations = relations[:max_num]
        instruction = instruction_template.render(sentences=sentences, relations=relations)
        response = response_template.render(level=coherence_levels[score])

        sample = {
            'id': f'{idx}',
            'system_prompt': 'You are an AI assistant tasked with coherence assessment.',
            'instruction': instruction,
            'output': response
        }
        new_data_list.append(sample)

    return new_data_list


def data_to_alpace_format_zero_sent(
    data_list,
    instruction_template,
    response_template
):
    new_data_list = []
    print(len(data_list))
    for idx, data in enumerate(data_list):
        text = data["text"]
        sents = data["sents"]
        score = data["score"]
        rels = data["rels"]
        spans = data["spans"]

        sentences = [(idx + 1, sent) for idx, sent in enumerate(sents)]
        instruction = instruction_template.render(sentences=sentences)
        response = response_template.render(level=coherence_levels[score])

        sample = {
            'id': f'{idx}',
            'system_prompt': 'You are an AI assistant tasked with coherence assessment.',
            'instruction': instruction,
            'output': response
        }
        new_data_list.append(sample)

    return new_data_list

def process_example(
    data,
    no_rel=False,
    no_enty=False,
    max_num=32
):
    sents = data["sents"]
    rels = data["rels"]
    spans = data["spans"]

    sentences = [(idx + 1, sent) for idx, sent in enumerate(sents)]
    relations = []
    for rel, span in zip(rels, spans):
        if span[0] == span[1]:
            continue

        if no_rel:
            if rel == "entity":
                relations.append((span[0] + 1, rel, span[1] + 1))
        elif no_enty:
            if rel != "entity":
                relations.append((span[0] + 1, rel, span[1] + 1))
        else:
            relations.append((span[0] + 1, rel, span[1] + 1))
    relations = relations[:max_num]

    return sentences, relations
