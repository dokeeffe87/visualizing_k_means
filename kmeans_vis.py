# Import modules

import io
import numpy as np
import pandas as pd
import pyLDAvis
from collections import Counter
from collections import namedtuple
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def _get_cluster_coordinates(centres, cluster_size,
    radius=5, embedding_method='tsne'):

    TopicCoordinates = namedtuple(
        'TopicCoordinates',
        'topic x y topics cluster Freq'.split()
    )

    n_clusters = centres.shape[0]

    if embedding_method == 'pca':
        coordinates = _coordinates_pca(centres)
    else:
        coordinates = _coordinates_tsne(centres)

    # scaling
    coordinates = 5 * coordinates / max(coordinates.max(), abs(coordinates.min()))

    cluster_size = np.asarray(
        [np.sqrt(cluster_size[c] + 1) for c in range(n_clusters)])
    cs_min, cs_max = cluster_size.min(), cluster_size.max()
    cluster_size = radius * (cluster_size - cs_min) / (cs_max - cs_min) + 0.2

    topic_coordinates = [
        TopicCoordinates(c+1, coordinates[i,0], coordinates[i,1], i+1, 1, cluster_size[c])
        for i, c in enumerate(sorted(range(n_clusters), key=lambda x:-cluster_size[x]))
    ]

    topic_coordinates = sorted(topic_coordinates, key=lambda x:-x.Freq)
    return topic_coordinates


def _coordinates_pca(centers):
    return PCA(n_components=2).fit_transform(centers)


def _coordinates_tsne(centers):
    return TSNE(n_components=2, metric='cosine').fit_transform(centers)


def _kmeans_to_prepared_data_pyldavis_score(x, index2word,
    centers, labels, embedding_method='tsne', radius=3.5,
    n_candidate_words=50, n_printed_words=30, lambda_step=0.01):
    """
    Dont use pyLDAvis embedding method. It shows unstable training results.
    """

    topic_term_dists = normalize(centers, norm='l1')

    empty_clusters = np.where(topic_term_dists.sum(axis=1) == 0)[0]
    default_weight = 1 / centers.shape[1]
    topic_term_dists[empty_clusters,:] = default_weight

    doc_topic_dists = np.zeros((x.shape[0], centers.shape[0]))
    for d, label in enumerate(labels):
        doc_topic_dists[d,label] = 1

    doc_lengths = x.sum(axis=1).A.ravel()

    term_frequency = x.sum(axis=0).A.ravel()
    term_frequency[term_frequency == 0] = 0.01 # preventing zeros

    if embedding_method == 'tsne':
        return pyLDAvis.prepare(
            topic_term_dists, doc_topic_dists, doc_lengths, index2word, term_frequency,
            R=radius, lambda_step=lambda_step, sort_topics=True,
            plot_opts={'xlab': 't-SNE1', 'ylab': 't-SNE2'}
        )
    else:
        return pyLDAvis.prepare(
            topic_term_dists, doc_topic_dists, doc_lengths, index2word, term_frequency,
            R=radius, lambda_step=lambda_step
        )


def _df_cluster_coordinate(cluster_coordinates):
    with io.open('./topic_coordinates.csv', 'w', encoding='utf-8') as f:
        f.write(u'topic,x,y,topics,cluster,Freq\n')
        for row in cluster_coordinates:
            row_strf = ','.join((str(v) for v in row))
            f.write(u'%s\n' % row_strf)
    return pd.read_csv('./topic_coordinates.csv')


def _df_cluster_info(cluster_info):
    with io.open('./topic_info.csv', 'w', encoding='utf-8') as f:
        f.write(u'term,Category,Freq,Term,Total,loglift,logprob\n')
        for row in cluster_info:
            row_strf = ','.join((str(v) for v in row))
            f.write(u'%s\n' % row_strf)
    return pd.read_csv('./topic_info.csv')


def _df_cluster_table(cluster_table):
    with io.open('./token_table.csv', 'w', encoding='utf-8') as f:
        f.write(u'term,Topic,Freq,Term\n')
        for row in cluster_table:
            row_strf = ','.join((str(v) for v in row))
            f.write(u'%s\n' % row_strf)
    return pd.read_csv('./token_table.csv')


def _get_cluster_table(weighted_centres, cluster_info, index2word):
    TokenTable = namedtuple('TokenTable', 'term Topic Freq Term'.split())

    term_proportion = weighted_centres / weighted_centres.sum(axis=0)

    token_table = []
    for info in cluster_info:
        try:
            c = int(info.Category[5:])
        except:
            # Category: Default
            continue
        token_table.append(
            TokenTable(
                info.term,
                c,
                term_proportion[c-1,info.term],
                info.Term
            )
        )

    return token_table


def _get_cluster_info(centres, cluster_size, index2word,
    weighted_centres, term_frequency, n_candidate_words=100):

    TopicInfo = namedtuple(
        'TopicInfo',
        'term Category Freq Term Total loglift logprob'.split()
    )

    l1_normalize = lambda x:x/x.sum()
    n_clusters, n_terms = centres.shape

    weighted_centre_sum = weighted_centres.sum(axis=0)
    total_sum = weighted_centre_sum.sum()
    term_proportion = weighted_centres / weighted_centre_sum

    topic_info = []

    # Category: Default
    default_terms = term_frequency.argsort()[::-1][:n_candidate_words]
    default_term_frequency = term_frequency[default_terms]
    default_term_loglift = 15 * default_term_frequency / default_term_frequency.max() + 10
    for term, freq, loglift in zip(default_terms, default_term_frequency, default_term_loglift):
        topic_info.append(
            TopicInfo(
                term,
                'Default',
                0.99,
                index2word[term],
                term_frequency[term],
                loglift,
                loglift
            )
        )

    # Category: for each topic
    for c, n_docs in enumerate(cluster_size):
        if n_docs == 0:
            keywords.append([])
            continue

        topic_idx = c + 1

        n_prop = l1_normalize(weighted_centre_sum - (centres[c] * n_docs))
        p_prop = l1_normalize(centres[c])

        indices = np.where(p_prop > 0)[0]
        indices = sorted(indices, key=lambda idx:-p_prop[idx])[:n_candidate_words]
        scores = [(idx, p_prop[idx] / (p_prop[idx] + n_prop[idx])) for idx in indices]

        for term, loglift in scores:
            topic_info.append(
                TopicInfo(
                    term,
                    'Topic%d' % topic_idx,
                    term_proportion[c, term] * term_frequency[term],
                    index2word[term],
                    term_frequency[term],
                    loglift,
                    p_prop[term]
                )
            )

    return topic_info