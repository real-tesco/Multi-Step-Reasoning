from pyserini.search import SimpleSearcher


class BM25Retriever:
    def __init__(self, dataset):
        self._searcher = SimpleSearcher(dataset)
        # self._searcher.set_bm25(3.44, 0.87)
        # self._searcher.set_rm3(10, 10, 0.5)

    def query(self, query_text, k=100):
        hits = self._searcher.search(query_text, k=k)
        return hits, query_text
