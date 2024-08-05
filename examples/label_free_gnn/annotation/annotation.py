from .at_helpers import *


def annotate(data, pl_indices, llm, n_tries=3):
    """
    Get predictions and confidences using LLM.
    """

    # Generate prompts
    prompts = generate_prompts(data.text, pl_indices, data.label_names, 3)

    # Query LLM
    responses = query_llm(llm, prompts, data.label_names, n=n_tries)

    # Process raw response and get answers
    res = collect_answers(responses, data.label_names)

    # Evaluate answers, get final results
    select_pred, select_conf = get_final_results(res, data.label_names, pl_indices, data.y)

    pred = -1 * torch.ones(data.num_nodes, dtype=torch.long)
    conf = torch.zeros(data.num_nodes, dtype=torch.float)
    pred[pl_indices] = select_pred
    conf[pl_indices] = select_conf
    data.pl = pred
    data.conf = conf

    return data
