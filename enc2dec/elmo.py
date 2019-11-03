from torch.nn import Module
from allennlp.modules.elmo import Elmo


class ElmoEmbedding(Module):
    def __init__(self, config):
        super(ElmoEmbedding, self).__init__()
        self.elmo = self._init_elmo(config)
        self.embedding_dim = 1024

    def _init_elmo(self, config):
        options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"  
        weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"   
        elmo_emb = Elmo(options_file, weight_file, 1)
        return elmo_emb

    def forward(self, input_tensor):
        return self.elmo(input_tensor)['elmo_representations'][0]

