import argparse
import numpy as np
import scipy.io
from collections import Counter
from topmost.evaluations import dynamic_TD, dynamic_TC

import sys
sys.path.append('./')
from utils.data import file_utils
import logging

import numpy as np
from gensim.topic_coherence import direct_confirmation_measure

log = logging.getLogger(__name__)

ADD_VALUE = 1


def custom_log_ratio_measure(segmented_topics, accumulator, normalize=False, with_std=False, with_support=False):
    topic_coherences = []
    num_docs = float(accumulator.num_docs)
    for s_i in segmented_topics:
        segment_sims = []
        for w_prime, w_star in s_i:
            w_prime_count = accumulator[w_prime]
            w_star_count = accumulator[w_star]
            co_occur_count = accumulator[w_prime, w_star]

            if normalize:
                # For normalized log ratio measure
                numerator = custom_log_ratio_measure([[(w_prime, w_star)]], accumulator)[0]
                co_doc_prob = co_occur_count / num_docs
                m_lr_i = numerator / (-np.log(co_doc_prob + direct_confirmation_measure.EPSILON))
            else:
                # For log ratio measure without normalization
                ### _custom: Added the following 6 lines, to prevent a division by zero error.
                if w_star_count == 0:
                    log.info(f"w_star_count of {w_star} == 0. Adding {ADD_VALUE} to the count to prevent error. ")
                    w_star_count += ADD_VALUE
                if w_prime_count == 0:
                    log.info(f"w_prime_count of {w_prime} == 0. Adding {ADD_VALUE} to the count to prevent error. ")
                    w_prime_count += ADD_VALUE
                numerator = (co_occur_count / num_docs) + direct_confirmation_measure.EPSILON
                denominator = (w_prime_count / num_docs) * (w_star_count / num_docs)
                m_lr_i = np.log(numerator / denominator)

            segment_sims.append(m_lr_i)

        topic_coherences.append(direct_confirmation_measure.aggregate_segment_sims(segment_sims, with_std, with_support))

    return topic_coherences
from gensim.topic_coherence import direct_confirmation_measure
# from my_custom_module import custom_log_ratio_measure

direct_confirmation_measure.log_ratio_measure = custom_log_ratio_measure

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--topic_path', type=str, required=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    train_times = np.loadtxt(f'{args.data_dir}/train_times.txt')
    train_texts = file_utils.read_text(f'{args.data_dir}/train_texts.txt')
    train_bow = scipy.sparse.load_npz(f'{args.data_dir}/train_bow.npz').toarray().astype('float32')
    train_times = np.loadtxt(f'{args.data_dir}/train_times.txt').astype('int32')
    vocab = file_utils.read_text(f'{args.data_dir}/vocab.txt')

    time_topic_dict = file_utils.read_topic_words(args.topic_path)

    time_idx = np.sort(np.unique(train_times))

    TC = dynamic_TC(train_texts, train_times, vocab, list(time_topic_dict.values()))
    print(f"===>dynamic_TC: {TC:.5f}")
    print(type(train_bow))
    TD = dynamic_TD(time_idx, time_topic_dict, train_bow, train_times, vocab)
    print(f"===>dynamic_TD: {TD:.5f}")
