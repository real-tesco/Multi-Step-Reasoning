import torch
import torch.nn.functional as F

class QueryReformulator:
    def __init__(self, mode: str):
        self._mode = mode

    def __call__(self, *args, **kwargs):
        if self._mode == 'top1':
            return self.replace_with_document(args)
        elif self._mode == 'top5':
            return self.replace_with_avg(args)

    def replace_with_document(self, document_vectors):
        print(type(document_vectors))
        print(document_vectors.shape)
        print(document_vectors[:, 5].shape)
        return document_vectors[:, :5]

    def replace_with_avg(self, document_vectors):
        rst = torch.mean(document_vectors[:, :5], dim=1)
        return rst