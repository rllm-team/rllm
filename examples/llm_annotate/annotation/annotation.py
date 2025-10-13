import torch

from .at_helpers import *
from examples.test_annotate.data import load_cache, save_cache


def annotate(name, data, mask, llm, use_cache, n_tries=1):
    """
    Get predictions and confidences using LLM.
    """
    print('Annotating {}...'.format(mask.sum()))

    pred = -1 * torch.ones(data.num_nodes, dtype=torch.long)
    conf = torch.zeros(data.num_nodes, dtype=torch.float)

    if use_cache:
        cached_annotation = load_cache(name)
        if cached_annotation:
            pred = torch.tensor(cached_annotation['pred'])
            conf = torch.tensor(cached_annotation['conf'])
            cached_mask = torch.tensor(cached_annotation['mask'])
            cached_indices = torch.nonzero(cached_mask, as_tuple=False).squeeze()
            pred[cached_indices] = pred[cached_indices]
            conf[cached_indices] = conf[cached_indices]

            hit_num = (mask & cached_mask).sum()
            print(f'cache hit num: {hit_num}')

            mask = mask & ~cached_mask
        else:
            cached_mask = mask & ~mask

    if mask.sum() == 0:
        return pred, conf

    select_indices = torch.nonzero(mask, as_tuple=False).squeeze()

    # Generate prompts
    prompts = generate_prompts(name, data.text, select_indices, data.label_names)

    # Query LLM
    responses = query_llm(llm, prompts, data.label_names, n=n_tries)

    # Process raw response and get answers
    res = collect_answers(responses, data.label_names)

    # Evaluate answers, get final results
    select_pred, select_conf = get_final_results(res, data.label_names, select_indices, data.y)

    pred[select_indices] = select_pred
    conf[select_indices] = select_conf

    if use_cache:
        save_cache(name, pred, mask | cached_mask, conf)
        print(f'{mask.sum()} updated to cache')

    return pred, conf
