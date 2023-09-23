import argparse
import json
import logging
import os
import random
import time
import torch
from datetime import timedelta
from nltk import pos_tag
from nltk.tree import Tree
import re

# Loading WIKIDATA client for getting EL
from wikidata.client import Client
client = Client()  # doctest: +SKIP


def _read_conll(path, encoding='utf-8', sep=None, indexes=None, dropna=True):
    r"""
    Construct a generator to read conll items.
    :param path: file path
    :param encoding: file's encoding, default: utf-8
    :param sep: seperator
    :param indexes: conll object's column indexes that needed, if None, all columns are needed. default: None
    :param dropna: weather to ignore and drop invalid data,
            :if False, raise ValueError when reading invalid data. default: True
    :return: generator, every time yield (line number, conll item)
    """

    def parse_conll(sample):

        # correct the number of columns (topres19th dev issue)
        sample = [item + ['O'] * len(indexes) for item in sample]
        sample = list(map(list, zip(*sample)))

        sample = [sample[i] for i in indexes]

        for f in sample:
            if len(f) <= 0:
                raise ValueError('empty field')
        return sample

    with open(path, 'r', encoding=encoding) as f:

        sample = []
        start = next(f).strip()  # Skip columns
        start = next(f).strip()

        data = []
        for line_idx, line in enumerate(f, 0):
            line = line.strip()

            if any(
                substring in line for substring in [
                    'DOCSTART',
                    '###',
                    "# id",
                    "# ",
                    '###']):
                continue

            if line == '':
                if len(sample):
                    try:
                        res = parse_conll(sample)
                        sample = []
                        if ['TOKEN'] not in res:
                            if ['Token'] not in res:
                                data.append([line_idx, res])
                    except Exception as e:
                        if dropna:
                            print(
                                'Invalid instance which ends at line: {} has been dropped.'.format(line_idx))
                            sample = []
                            raise e
                        raise ValueError(
                            'Invalid instance which ends at line: {}'.format(line_idx))
            elif 'EndOfSentence' in line:
                sample.append(
                    line.split(sep)) if sep else sample.append(
                    line.split())

                if len(sample):
                    try:
                        res = parse_conll(sample)
                        sample = []
                        if ['TOKEN'] not in res:
                            if ['Token'] not in res:
                                data.append([line_idx, res])
                    except Exception as e:
                        if dropna:
                            print(
                                'Invalid instance which ends at line: {} has been dropped.'.format(line_idx))
                            sample = []
                            raise e
                        raise ValueError(
                            'Invalid instance which ends at line: {}'.format(line_idx))
            else:
                sample.append(
                    line.split(sep)) if sep else sample.append(
                    line.split())

        if len(sample) > 0:
            try:
                res = parse_conll(sample)
                if ['TOKEN'] not in res:
                    if ['Token'] not in res:
                        data.append([line_idx, res])
            except Exception as e:
                if dropna:
                    return
                print('Invalid instance ends at line: {}'.format(line_idx))
                raise e

        return data


class LogFormatter():

    def __init__(self):
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = round(record.created - self.start_time)

        prefix = "%s - %s - %s" % (
            record.levelname,
            time.strftime("%x %X"),
            timedelta(seconds=elapsed_seconds)
        )
        message = record.getMessage()
        message = message.replace('\n', '\n' + ' ' * (len(prefix) + 3))
        return "%s - %s" % (prefix, message) if message else ''


def conlltags2tree(sentence, chunk_types=('NP', 'PP', 'VP'),
                   root_label='S', strict=False):
    """
    Convert the CoNLL IOB format to a tree.
    """
    tree = Tree(root_label, [])
    for (word, chunktag, idx) in sentence:
        idx = str(idx)
        if chunktag is None:
            if strict:
                raise ValueError("Bad conll tag sequence")
            else:
                # Treat as O
                tree.append((word, idx))
        elif chunktag.startswith('B-'):
            tree.append(Tree(chunktag[2:], [(word, idx)]))
        elif chunktag.startswith('I-'):
            if len(tree) == 0 or not isinstance(
                    tree[-1], Tree) or tree[-1].label() != chunktag[2:]:
                if strict:
                    raise ValueError("Bad conll tag sequence")
                else:
                    # Treat as B-*
                    tree.append(Tree(chunktag[2:], [(word, idx)]))
            else:
                tree[-1].append((word, idx))
        elif chunktag == 'O':
            tree.append((word, idx))
        else:
            print(word, idx)
            raise ValueError("Bad conll tag %r" % chunktag)
    return tree


def get_entities(tokens, entity_tags, link_tags):
    # print(tokens, entity_tags, link_tags)
    entity_tags = [tag.replace('S-', 'B-').replace('E-', 'I-')
                   for tag in entity_tags]
    link_tags = [entity_tags[idx][:2] + tag if tag !=
                 '_' else 'O' for idx, tag in enumerate(link_tags)]
    pos_tags = [pos for token, pos in pos_tag(tokens)]

    indices = list(range(len(entity_tags)))
    entity_tags = [
        (token, tag, idx) for token, pos, tag, idx in zip(
            tokens, pos_tags, entity_tags, indices)]
    link_tags = [(token, tag, idx) for token, pos, tag,
                 idx in zip(tokens, pos_tags, link_tags, indices)]

    ne_tree = conlltags2tree(entity_tags)
    el_tree = conlltags2tree(link_tags)

    entities = []
    for ne_subtree, el_subtree in zip(ne_tree, el_tree):
        # skipping 'O' tags
        # print(ne_subtree, '---', el_subtree)
        if isinstance(ne_subtree, Tree):
            en_label = ne_subtree.label()
            if not isinstance(
                    el_subtree,
                    str) and not isinstance(
                    el_subtree,
                    tuple):
                el_label = el_subtree.label()
            else:
                el_label = 'NIL'
            en_string = " ".join([token[0] for token in ne_subtree.leaves()])
            en_idx = [int(token[1]) for token in ne_subtree.leaves()]
            entities.append((en_string, en_label, el_label, en_idx))
    return entities
