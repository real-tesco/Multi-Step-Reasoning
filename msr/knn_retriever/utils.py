import torch


def batchify(args, para_mode, train_time):
    return lambda x: batchify_(args, x, para_mode, train_time)


def batchify_(args, batch, para_mode, train_time):
    """Gather a batch of individual examples into one batch."""

    new_batch = []
    for d in batch:
        if d is not None:
            new_batch.append(d)
    batch = new_batch
    if len(batch) == 0:
        return None
    ids = [ex[-1] for ex in batch]
    docs = [ex[0] for ex in batch]
    questions = [ex[1] for ex in batch]
    num_occurances = [ex[-2] for ex in batch]
    num_occurances = torch.LongTensor(num_occurances)
    # Batch documents and features
    max_length = max([d.size(0) for d in docs])
    x1 = torch.LongTensor(len(docs), max_length).zero_()
    x1_mask = torch.ByteTensor(len(docs), max_length).fill_(1)

    for i, d in enumerate(docs):
        x1[i, :d.size(0)].copy_(d)
        x1_mask[i, :d.size(0)].fill_(0)

    # Batch questions
    max_length = max([q.size(0) for q in questions])
    x2 = torch.LongTensor(len(questions), max_length).zero_()
    x2_mask = torch.ByteTensor(len(questions), max_length).fill_(1)
    for i, q in enumerate(questions):
        x2[i, :q.size(0)].copy_(q)
        x2_mask[i, :q.size(0)].fill_(0)

    return x1, x1_mask, x2, x2_mask, num_occurances, ids