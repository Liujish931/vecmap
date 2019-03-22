# Copyright (C) 2016-2018  Mikel Artetxe <artetxem@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
import collections
import torch

from cupy_utils import *
from elmoformanylangs import Embedder
import numpy as np
try:
    import cupy
except ImportError:
    cupy = None

BATCH_SIZE = 1000


def topk_mean(m, k, inplace=False, axis=1):
    n = m.shape[0]
    ans = xp.zeros(n, dtype=m.dtype)
    if k <= 0:
        return ans
    if not inplace:
        m = xp.array(m)
    ind0 = xp.arange(n)
    ind1 = xp.empty(n, dtype=int)
    minimum = m.min()
    for i in range(k):
        m.argmax(axis=axis, out=ind1)
        ans += m[ind0, ind1]
        m[ind0, ind1] = minimum
    return ans / k


def read(file, threshold=0, vocabulary=None, dtype='float'):
    header = file.readline().split(' ')
    count = int(header[0]) if threshold <= 0 else min(threshold, int(header[0]))
    dim = int(header[1])
    words = []
    matrix = np.empty((count, dim), dtype=dtype) if vocabulary is None else []
    for i in range(count):
        word, vec = file.readline().split(' ', 1)
        if vocabulary is None:
            words.append(word)
            matrix[i] = np.fromstring(vec, sep=' ', dtype=dtype)
        elif word in vocabulary:
            words.append(word)
            matrix.append(np.fromstring(vec, sep=' ', dtype=dtype))
    return (words, matrix) if vocabulary is None else (words, np.array(matrix, dtype=dtype))


def write(words, matrix, file):
    m = asnumpy(matrix)
    print('%d %d' % m.shape, file=file)
    for i in range(len(words)):
        print(words[i] + ' ' + ' '.join(['%.6g' % x for x in m[i]]), file=file)


def length_normalize(matrix):
    xp = get_array_module(matrix)
    norms = xp.sqrt(xp.sum(matrix**2, axis=1))
    norms[norms == 0] = 1
    matrix /= norms[:, xp.newaxis]
    # print('matrix sum = ', xp.sum(matrix))


def mean_center(matrix):
    xp = get_array_module(matrix)
    avg = xp.mean(matrix, axis=0)
    matrix -= avg
    # print(' mean center = ', xp.sum(avg), ' shape ', xp.shape(avg))
    # print('matrix sum = ', xp.sum(matrix), ' shape ', xp.shape(matrix))
    # print(matrix)
    return avg


def length_normalize_dimensionwise(matrix):
    xp = get_array_module(matrix)
    norms = xp.sqrt(xp.sum(matrix**2, axis=0))
    norms[norms == 0] = 1
    matrix /= norms


def mean_center_embeddingwise(matrix):
    xp = get_array_module(matrix)
    avg = xp.mean(matrix, axis=1)
    matrix -= avg[:, xp.newaxis]


def normalize(matrix, actions):
    avg = 0
    for action in actions:
        if action == 'unit':
            length_normalize(matrix)
        elif action == 'center':
            avg = mean_center(matrix)
        elif action == 'unitdim':
            length_normalize_dimensionwise(matrix)
        elif action == 'centeremb':
            mean_center_embeddingwise(matrix)
    if 'center' in actions:
        return avg


def nearest_neighbour(word, x, id2word, word2id, topn=20, retrieval='cos'):

    if type(word) == str and word not in id2word:
        raise ValueError(word, 'is not in the vocab')

    # normalize(x, ['unit', 'center', 'unit'])
    normalize(x, ['unit'])

    if type(word) == str:
        word_id = word2id[word]
        word_vec = x[word_id]
    else:
        word_vec = word

    last_batch_similarities = None
    if retrieval == 'cos':
        for i in range(0, len(id2word), BATCH_SIZE):
            j = min(i + BATCH_SIZE, len(id2word))
            similarities = x[i:j].dot(word_vec.T)
            if last_batch_similarities is not None:
                similarities = xp.concatenate((last_batch_similarities, similarities))

            last_batch_similarities = similarities
    elif retrieval == 'csls':  # Cross-domain similarity local scaling
        knn_sim_bwd = xp.zeros(x.shape[0])
        for i in range(0, x.shape[0], BATCH_SIZE):
            j = min(i + BATCH_SIZE, x.shape[0])
            knn_sim_bwd[i:j] = topk_mean(x[i:j].dot(x.T), k=10, inplace=True, axis=1)
            # print(knn_sim_bwd[i:j].shape)
        for i in range(0, len(id2word), BATCH_SIZE):
            j = min(i + BATCH_SIZE, len(id2word))
            similarities = 2*x[i:j].dot(word_vec.T) - knn_sim_bwd[i:j]  # Equivalent to the real CSLS scores for NN
            # nn = similarities.argmax(axis=1).tolist()
            if last_batch_similarities is not None:
                similarities = xp.concatenate((last_batch_similarities, similarities))
            # print(similarities.shape)
            last_batch_similarities = similarities

    if type(word) == str:
        similarities[word_id] = float("-inf")
    l = similarities.argsort(axis=0).tolist()
    l.reverse()
    print(l[:topn])
    return [(id2word[i], similarities[i]) for i in l[:topn]]


def map_matrix(x, w1, w2, s=None):
    # whitening
    x = x.dot(w1)
    # othogonal mapping
    x = x.dot(w2)
    # reweighting
    if s is not None:
        x *= s ** 0.5
    # dewhitening
    x = x.dot(w2.T.dot(xp.linalg.inv(w1)).dot(w2))
    return x


if __name__ == '__main__':
    # cloud setting
    # DATA_PATH = '/home/ljingshu/dev/jingshu/vecamp_data/data/'
    # DATA_PATH = '/home/ubuntu/vecamp_data/data/'
    # dicta setting
    DATA_PATH = 'data/'

    # Read input embeddings
    srcfile_name = DATA_PATH + 'embeddings/original/ELMo/en.txt'
    # srcfile_name = DATA_PATH + 'embeddings/mapped/ELMo/en-fr/en.txt'

    # wx1
    src_whitening_matrix_filename = DATA_PATH + 'embeddings/mapped/ELMo/en-fr/en.txt_whitening_matrix'
    # wx2
    src_matrix_u_filename = DATA_PATH + 'embeddings/mapped/ELMo/en-fr/en.txt_othogonal_mapping_u'


    # wz1
    trg_whitening_matrix_filename = DATA_PATH + 'embeddings/mapped/ELMo/en-fr/fr.txt_whitening_matrix'
    # wz2
    trg_matrix_v_filename = DATA_PATH + 'embeddings/mapped/ELMo/en-fr/fr.txt_othogonal_mapping_v'

    # s
    s_matrix_filename = DATA_PATH + 'embeddings/mapped/ELMo/en-fr/en.txt_s'

    trgfile_name = DATA_PATH + 'embeddings/original/ELMo/fr.txt'
    # trgfile_name = DATA_PATH + 'embeddings/mapped/ELMo/en-fr/fr.txt'

    srcfile = open(srcfile_name, errors='surrogateescape')
    trgfile = open(trgfile_name, errors='surrogateescape')
    # src_words and trg_words are word2id list
    src_words, x = read(srcfile)
    trg_words, z = read(trgfile)

    wx1 = np.loadtxt(src_whitening_matrix_filename)
    wx2 = np.loadtxt(src_matrix_u_filename)

    wz1 = np.loadtxt(trg_whitening_matrix_filename)
    wz2 = np.loadtxt(trg_matrix_v_filename)

    s = np.loadtxt(s_matrix_filename)

    # Build word to index map
    src_word2ind = {word: i for i, word in enumerate(src_words)}
    trg_word2ind = {word: i for i, word in enumerate(trg_words)}

    # NumPy/CuPy management
    if supports_cupy():
        xp = get_cupy()
        x = xp.asarray(x)
        wx1 = xp.asarray(wx1)
        wx2 = xp.asarray(wx2)
        wz1 = xp.asarray(wz1)
        wz2 = xp.asarray(wz2)
        s = xp.asarray(s)
        z = xp.asarray(z)
    else:
        xp = np

    print(x.sum())
    print(z.sum())

    print(x[src_word2ind['bank']].sum())

    avg_x = normalize(x, ['unit', 'center', 'unit'])
    avg_z = normalize(z, ['unit', 'center', 'unit'])

    print('dimension wise mean of source matrix =', avg_x, avg_x.shape)
    print('dimension wise mean of target matrix =', avg_z, avg_z.shape)

    # normalize(x, ['unit'])
    x = map_matrix(x, wx1, wx2, s=s)
    z = map_matrix(z, wz1, wz2, s=s)

    print(x.sum())
    print(z.sum())

    word_vec = x[src_word2ind['wind']]
    res = nearest_neighbour(word_vec, z, trg_words, trg_word2ind, retrieval='cos')
    print(res)
    with torch.no_grad():
        e = Embedder('/home/jingshu/dev/jingshu/ELMoForManyLangs/model/en', batch_size=1000)
        e.model.eval()
        # sents = [['He', 'went', 'to', 'the', 'bank', 'to', 'save', 'some', 'money', '.']]
        # sents = [['In', 'geography',',', 'the', 'word', 'bank', 'generally', 'refers', 'to', 'the', 'land', 'alongside', 'a', 'body', 'of', 'water','.']]
        sents = [['bank']]
        # word_vec_in_context is a a numpy array
        word_vec_in_context = e.sents2elmo(sents)[0][0]
        print(word_vec_in_context.shape)
        print(word_vec_in_context.sum())

        # Equivalent
        # word_vec_in_context = np.expand_dims(word_vec_in_context, axis=0)
        word_vec_in_context = word_vec_in_context.reshape((1, word_vec_in_context.shape[0]))

        print(word_vec_in_context.shape)

        # unit length normalisation
        length_normalize(word_vec_in_context)
        # mean centering
        word_vec_in_context -= avg_x
        #  unit length normalisation
        length_normalize(word_vec_in_context)

        word_vec_in_context = map_matrix(word_vec_in_context, wx1, wx2, s=s)

        word_vec_in_context = np.squeeze(word_vec_in_context)

        res = nearest_neighbour(word_vec_in_context, z, trg_words, trg_word2ind, retrieval='cos')
        # res = nearest_neighbour(word_vec_in_context, x, src_words, src_word2ind, retrieval='cos')

        print(res)
