from random import shuffle, seed
import sys
import os.path
import argparse
import numpy as np
import scipy.io
import pdb
import h5py
from nltk.tokenize import word_tokenize
import json
import re
import math
import pdb

def prepro_question(data, params):
    # preprocess all the question
    print 'example processed tokens:'
    for i,img in enumerate(data):
        s = img['question'].encode("utf-8")
        if params['token_method'] == 'nltk':
            txt = word_tokenize(str(s).lower())
        else:
            txt = tokenize(s)

        img['processed_tokens'] = txt
        if i < 10: print txt
        if i % 1000 == 0:
            sys.stdout.write("processing %d/%d (%.2f%% done)   \r" %  (i, len(data), i*100.0/len(data)) )
            sys.stdout.flush()
    #pdb.set_trace()
    return data


def build_vocab_question(data, params):
    # build vocabulary for question and answers.
    count_thr = params['word_count_threshold']
    # count up the number of words
    counts = {}
    for d in data:
        for w in d['processed_tokens']:
            counts[w] = counts.get(w, 0) + 1
    cw = sorted([(count,w) for w,count in counts.iteritems()], reverse=True)
    print 'top words and their counts:'
    print '\n'.join(map(str,cw[:20]))

    # print some stats
    total_words = sum(counts.itervalues())
    print 'total words:', total_words
    bad_words = [w for w,n in counts.iteritems() if n <= count_thr]
    vocab = [w for w,n in counts.iteritems() if n > count_thr]
    bad_count = sum(counts[w] for w in bad_words)
    print 'number of bad words: %d/%d = %.2f%%' % (len(bad_words), len(counts), len(bad_words)*100.0/len(counts))
    print 'number of words in vocab would be %d' % (len(vocab), )
    print 'number of UNKs: %d/%d = %.2f%%' % (bad_count, total_words, bad_count*100.0/total_words)


    # lets now produce the final annotation
    # additional special UNK token we will use below to map infrequent words
    print 'inserting the special UNK token'
    vocab.append('UNK')
    print 'inserting the special __PAD__ token'
    vocab.append('__PAD__')
    for d in data:
        txt = d['processed_tokens']
        question = [w if counts.get(w,0) > count_thr else 'UNK' for w in txt]
        d['final_question'] = question
    return data, vocab

def build_answer_vocab(d,params):
    counts = {}
    for inst in d:
        ans = inst['answer']
        counts[ans] = counts.get(ans,0)+1
    vocab = [w for w,n in counts.iteritems()]
    return vocab

def apply_vocab_question(imgs, wtoi):
    # apply the vocab on test.
    for img in imgs:
        txt = img['processed_tokens']
        question = [w if w in wtoi else 'UNK' for w in txt]
        img['final_question'] = question

    return imgs

def main(params):
#    data_trn = [json.loads(line) for line in open(params['input_train_json'])]
#    data_dev = [json.loads(line) for line in open(params['input_dev_json'])]
#    data_tst = [json.loads(line) for line in open(params['input_test_json'])]

    data_trn = json.load(open(params['input_train_json']))['questions']
    data_dev = json.load(open(params['input_dev_json']))['questions']
    data_tst = json.load(open(params['input_test_json']))['questions']


    data_trn = prepro_question(data_trn,params)
    data_dev = prepro_question(data_dev,params)
    data_tst = prepro_question(data_tst,params)
    data_trn,vocab = build_vocab_question(data_trn,params)
    itow = {i+1:w for i,w in enumerate(vocab)}
    wtoi = {w:i+1 for i,w in enumerate(vocab)}
    vocabs = {}
    vocabs['i2w'] = itow
    vocabs['w2i'] = wtoi
    data_tst = apply_vocab_question(data_tst, wtoi)
    data_dev = apply_vocab_question(data_dev, wtoi)
    #pdb.set_trace()
    avocab = build_answer_vocab(data_trn,params)
    atoi = {w:i+1 for i,w in enumerate(avocab)}

    trn_processed = os.path.join(params['outputdir'],'train_processed.json')
    with open(trn_processed,'w') as outfile:
        json.dump(data_trn,outfile)

    dev_processed = os.path.join(params['outputdir'],'val_processed.json')
    with open(dev_processed,'w') as outfile:
        json.dump(data_dev,outfile)

    tst_processed = os.path.join(params['outputdir'],'test_processed.json')
    with open(tst_processed,'w') as outfile:
        json.dump(data_tst,outfile)

    vocabularyp = os.path.join(params['outputdir'],'vocabulary.json')
    with open(vocabularyp,'w') as outfile:
        json.dump(vocabs,outfile)

    vocabularya= os.path.join(params['outputdir'],'answer_vocabulary.json')
    with open(vocabularya,'w') as outfile:
        json.dump(atoi,outfile)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_train_json', default='CLEVR_train_questions.json', help='train json file ')
    parser.add_argument('--input_dev_json', default='CLEVR_val_questions.json', help='dev json file')
    parser.add_argument('--input_test_json', default='CLEVR_test_questions.json', help='test json file')
    parser.add_argument('--outputdir', default='../data/processed', help='path for processed files')
    parser.add_argument('--word_count_threshold', default=0, type=int, help='only words that occur more than this number of times will be put in vocab')
    parser.add_argument('--token_method', default='nltk', help='token method, nltk is much more slower.')
    args = parser.parse_args()
    params = vars(args) # convert to ordinary dict
    print 'parsed input parameters:'
    print json.dumps(params, indent = 2)
    main(params)
