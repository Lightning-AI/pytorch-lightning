import torch
import numpy as np
from copy import deepcopy


class PretrainedEmbedding(torch.nn.Embedding):

    def __init__(self, embedding_path, embedding_dim, task_vocab, freeze=True, *args, **kwargs):
        """
        Loads a prebuilt pytorch embedding from any embedding formated file.
        Padding=0 by default.

        >>> emb = PretrainedEmbedding(embedding_path='glove.840B.300d.txt',embedding_dim=300, task_vocab={'hello': 1, 'world': 2})
        >>> data = torch.Tensor([[0, 1], [0, 2]]).long()
        >>> embedded = emb(data)
        


        :param embedding_path:
        :param emb_dim:
        :param task_vocab:
        :param freeze:
        :return:
        """
        # count the vocab
        self.vocab_size = max(task_vocab.values()) + 1
        super(PretrainedEmbedding, self).__init__(self.vocab_size, embedding_dim, padding_idx=0, *args, **kwargs)

        # load pretrained embeddings
        new_emb = self.__load_task_specific_embeddings(deepcopy(task_vocab), embedding_path, embedding_dim, freeze)

        # transfer weights
        self.weight = new_emb.weight

        # apply freeze
        should_freeze = not freeze
        self.weight.requires_grad = should_freeze

    def __load_task_specific_embeddings(self, vocab_words, embedding_path, emb_dim, freeze):
        """
        Iterates embedding file to only pull out task specific embeddings
        :param vocab_words:
        :param embedding_path:
        :param emb_dim:
        :param freeze:
        :return:
        """

        # holds final embeddings for relevant words
        embeddings = np.zeros(shape=(self.vocab_size, emb_dim))

        # load embedding line by line and extract relevant embeddings
        with open(embedding_path, encoding='utf-8') as f:
            for line in f:
                tokens = line.split(' ')
                word = tokens[0]
                embedding = tokens[1:]
                embedding[-1] = embedding[-1][:-1]  # remove last new line

                if word in vocab_words:
                    vocab_word_i = vocab_words[word]

                    # skip words that try to overwrite pad idx
                    if vocab_word_i == 0:
                        del vocab_words[word]
                        continue

                    emb_vals = np.asarray([float(x) for x in embedding])
                    embeddings[vocab_word_i] = emb_vals

                    # remove vocab word to early terminate
                    del vocab_words[word]

                # early break
                if len(vocab_words) == 0:
                    break

        # add random vectors for the non-pretrained words
        # these are vocab words NOT found in the pretrained embeddings
        for w, i in vocab_words.items():
            # skip words that try to overwrite pad idx
            if i == 0:
                continue

            embedding = np.random.normal(size=emb_dim)
            embeddings[i] = embedding

        # turn into pt embedding
        embeddings = torch.FloatTensor(embeddings)
        embeddings = torch.nn.Embedding.from_pretrained(embeddings, freeze=freeze)

        return embeddings


if __name__ == '__main__':
    emb = PretrainedEmbedding(
        embedding_path='/Users/waf/Developer',
        embedding_dim=300,
        task_vocab={'hello': 1, 'world': 2}
    )

    data = torch.Tensor([[0, 1], [0, 2]]).long()
    embedded = emb(data)
    print(embedded)
