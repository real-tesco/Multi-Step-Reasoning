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
        self.index.load_index(args.index)
        self.query_transformer = QueryTransformer(args)
        self.document_transformer = DocumentTransformer(args)
        if args.state_dict is not None:
            if 'q_transformer' in args.state_dict:
                self.query_transformer.load_state_dict(args.state_dict['q_transformer'])
            if 'd_transformer' in args.state_dict:
                self.document_transformer.load_state_dict(args.state_dict['d_transformer'])
            #self.query_transformer.eval()
        self.init_optimizer()

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
        scores_positive = queries * positives.transpose()
        scores_negative = queries * negatives.transpose()
        return queries, scores_positive, scores_negative

    #check if works, else pid needs to be N dim np array
    def get_passage(self, pid):
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
        return doc


class QueryTransformer(nn.Module):
    def __init__(self, args):
        super(QueryTransformer, self).__init__()
        self.linear_layer = nn.Linear(args.dim, args.dim)

    def forward(self, query):
        query = self.linear_layer(query)
        return query
