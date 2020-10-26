import hnswlib
import torch
import torch.nn as nn
import logging
import torch.optim as optim
import copy

logger = logging.getLogger()


class KnnIndex:
    def __init__(self, args):
        self.args = args
        self.index = hnswlib.Index(space=args.similarity, dim=args.dim)
        logger.info('Loading KNN index...')
        if not args.train:
            self.index.load_index(args.index)
        self.query_transformer = QueryTransformer(args)
        self.document_transformer = DocumentTransformer(args)
        if args.cuda:
            self.query_transformer.cuda()
            self.document_transformer.cuda()

        if args.state_dict is not None:
            if 'q_transformer' in args.state_dict:
                self.query_transformer.load_state_dict(args.state_dict['q_transformer'])
            if 'd_transformer' in args.state_dict:
                self.document_transformer.load_state_dict(args.state_dict['d_transformer'])
            #self.query_transformer.eval()
        #self.init_optimizer()

    def knn_query(self, query, k=1):
        query = self.query_transformer.forward(query)
        labels, distances = self.index.knn_query(query=query, k=k)
        return labels, distances

    def get_trainable_parameters(self):
        params = []
        for p in self.query_transformer.parameters():
            if p.requires_grad:
                params.append(p)
        for p in self.document_transformer.parameters():
            if p.requires_grad:
                params.append(p)
        return params

    #dont use optimizer in model, stick to external optimizer
    def init_optimizer(self):
        """Initialize an optimizer for the free parameters of the Query transformer.
        """

        parameters = self.get_trainable_parameters()

        if self.args.optimizer == 'sgd':
            self.optimizer = optim.SGD(parameters, self.args.learning_rate,
                                       momentum=self.args.momentum,
                                       weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'adamax':
            self.optimizer = optim.Adamax(parameters,
                                          weight_decay=self.args.weight_decay)
        else:
            raise RuntimeError('Unsupported optimizer: %s' %
                               self.args.optimizer)

    def save(self, filename):
        state_dict = {'q_transformer': copy.copy(self.query_transformer.state_dict()),
                      'd_transformer': copy.copy(self.document_transformer.state_dict())}
        #for document

        params = {
            'state_dict': state_dict,
            'args': self.args,
        }
        try:
            torch.save(params, filename)
            logger.info('Model saved at {}'.format(filename))
        except BaseException:
            logger.warning('WARN: Saving failed... continuing anyway.')

    def checkpoint(self, filename, epoch):
        state_dict = {'q_transformer': copy.copy(self.query_transformer.state_dict()),
                      'd_transformer': copy.copy(self.document_transformer.state_dict())}
        # for document
        params = {
            'state_dict': state_dict,
            'args': self.args,
            'epoch': epoch
        }
        try:
            torch.save(params, filename)
            logger.info('Model saved at {}'.format(filename))
        except BaseException:
            logger.warning('WARN: Saving failed... continuing anyway.')

    @staticmethod
    def load(filename):
        logger.info('Loading model %s' % filename)
        saved_params = torch.load(
            filename, map_location=lambda storage, loc: storage
        )
        state_dict = saved_params['state_dict']
        args = saved_params['args']
        args.state_dict = state_dict

        return KnnIndex(args)

    def score_documents(self, queries, positives, negatives):
        queries = self.query_transformer.forward(queries)
        positives = self.document_transformer.forward(positives)
        negatives = self.document_transformer.forward(negatives)

        #logger.info((queries[0] ** 2).sum())
        #logger.info((positives[0] ** 2).sum())

        scores_positive = queries * positives
        logger.info(scores_positive.shape)
        scores_positive = scores_positive.sum(dim=1)
        logger.info(scores_positive.shape)
        logger.info(scores_positive)
        scores_positive1 = torch.matmul(queries, positives.t())
        logger.info(scores_positive1.shape)
        logger.info(scores_positive1)

        scores_negative = queries * negatives
        scores_negative = scores_negative.sum(dim=1)

        '''
        scores_positive = torch.matmul(queries, positives.t())
        
        p = torch.FloatTensor(queries.shape[0])
        for idx, _ in enumerate(scores_positive.split(queries.shape[0], 0)):
            p[idx].copy_(scores_positive[idx][idx])

        negatives = self.document_transformer.forward(negatives)
        scores_negative = torch.matmul(queries, negatives.t())
        n = torch.FloatTensor(queries.shape[0])
        for idx, _ in enumerate(scores_negative.split(queries.shape[0], 0)):
            n[idx].copy_(scores_negative[idx][idx])
        logger.info(p)
        logger.info(p.requires_grad)
        logger.info(n.requires_grad)
        '''
        return scores_positive, scores_negative

    def get_passage(self, pid):
        # check if works, else pid needs to be N dim np array
        return self.index.get_items(pid)

    '''@staticmethod
    def load_checkpoint(filename, new_args, normalize=True):
        logger.info('Loading model %s' % filename)
        saved_params = torch.load(
            filename, map_location=lambda storage, loc: storage
        )
        word_dict = saved_params['word_dict']
        feature_dict = saved_params['feature_dict']
        state_dict = saved_params['state_dict']
        epoch = saved_params['epoch']
        optimizer = saved_params['optimizer']
        args = saved_params['args']
        if new_args:
            args = override_model_args(args, new_args)
        model = Model(args, word_dict, feature_dict, state_dict=state_dict, normalize=normalize)
        model.init_optimizer(optimizer)
        return model, epoch
        '''


class DocumentTransformer(nn.Module):
    def __init__(self, args):
        super(DocumentTransformer, self).__init__()
        self.linear_layer = nn.Linear(args.dim, args.dim)

    def forward(self, document):
        doc = self.linear_layer(document)
        #logger.info(f"doc before norm: {doc}")
        doc = nn.functional.normalize(doc, p=2, dim=1)
        #logger.info(f"doc after norm: {doc}")
        return doc


class QueryTransformer(nn.Module):
    def __init__(self, args):
        super(QueryTransformer, self).__init__()
        self.linear_layer = nn.Linear(args.dim, args.dim)

    def forward(self, query):
        query = self.linear_layer(query)
        query = nn.functional.normalize(query, p=2, dim=1)
        return query
