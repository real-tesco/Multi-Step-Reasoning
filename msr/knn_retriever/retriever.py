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
        logger.info('Loadingg KNN index...')
        self.index.load_index(args.index)
        self.query_transformer = QueryTransformer(args)
        if args.state_dict is not None:
            self.query_transformer.load_state_dict(args.state_dict)
            #self.query_transformer.eval()
        self.init_optimizer()

    def knn_query(self, query, k=1):
        query = self.query_transformer.forward(query)
        labels, distances = self.index.knn_query(query=query, k=k)
        return labels, distances

    def get_trainable_parameters(self):
        return [p for p in self.query_transformer.parameters() if p.requires_grad]

    def init_optimizer(self):
        """Initialize an optimizer for the free parameters of the Query transformer.
        """

        if self.query_transformer is not None:
            parameters = [p for p in self.query_transformer.parameters() if p.requires_grad]

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
        state_dict = copy.copy(self.query_transformer.state_dict())

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
        state_dict = copy.copy(self.query_transformer.state_dict())
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
        scores_positive = queries * positives
        scores_negative = queries * negatives
        return queries, scores_positive, scores_negative

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


class QueryTransformer(nn.Module):
    def __init__(self, args):
        super(QueryTransformer, self).__init__()
        self.linear_layer = nn.Linear(args.dim, args.dim)

    def forward(self, query):
        query = self.linear_layer(query)
        return query
