import torch

from rllm.utils import seg_softmax, seg_softmax_


def test_seg_softmax():
    data = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=torch.float)
    segment_ids = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    num_segs = 2
    score = seg_softmax(data, segment_ids, num_segs)
    score_ = seg_softmax_(data, segment_ids, num_segs)
    assert torch.equal(score, score_)
