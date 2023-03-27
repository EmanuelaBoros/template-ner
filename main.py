# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
import pickle
import re
# for pytorch/fairseq

from data_utils import _read_conll, get_entities
from tqdm import tqdm
import pandas as pd


def get_item(tokens, entity_tags, link_tags):
    entities = get_entities(tokens, entity_tags, link_tags)

    item = {"input_text": ' '.join(tokens)}
    # [('R . Ellis', 'pers', 'Q7344037', [12, 13, 14])
    pos_qid, pos_ner = {}, {}
    target_tokens = []
    for entity in entities:

        positions = entity[-1]
        entity_type = entity[1]
        entity_text = entity[0]
        target_tokens.append(entity_text + " [" + entity_type + "]")

        # if len(positions) == 1:
        #     tokens[positions[0]] = "[" + entity_type + "] " + entity_text + " [/" + entity_type + "]"
        # else:
        #     tokens[positions[0]] = "[" + entity_type + "] " + tokens[positions[0]]
        #     tokens[positions[-1]] = tokens[positions[-1]] + " [/" + entity_type + "]"

    item["target_text"] = ' '.join(target_tokens)
    return item


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_dir",
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
    )
    parser.add_argument(
        "--base_wikidata",
        type=str,
        help="Base folder with Wikidata data.",
    )
    parser.add_argument(
        "-d",
        "--debug",
        help="Print lots of debugging statements",
        action="store_const",
        dest="loglevel",
        const=logging.DEBUG,
        default=logging.WARNING,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Be verbose",
        action="store_const",
        dest="loglevel",
        const=logging.INFO,
    )

    args, _ = parser.parse_known_args()

    logging.basicConfig(level=logging.DEBUG)


for root, dirs, files in os.walk(os.path.join(args.input_dir), topdown=False):
    # print(files)
    for name in files:
        filename = os.path.join(root, name)
        if ('.tsv' in filename) \
                and ('masked' not in filename) \
                and ('results' not in filename) \
                and ('_nel' not in filename) \
                and ('bart' not in filename):

            logging.info("Converting {}".format(filename))
            with open(filename, 'r') as f:
                lines = f.readlines()

            headers = [
                'raw_words', 'target', 'link'
            ]
            # TODO: This needs to be changed if the data format is different or the
            # order of the elements in the file is different
            indexes = list(range(10))  # -3 is for EL
            columns = ["TOKEN", "NE-COARSE-LIT", "NE-COARSE-METO", "NE-FINE-LIT",
                       "NE-FINE-METO", "NE-FINE-COMP", "NE-NESTED",
                       "NEL-LIT", "NEL-METO", "MISC"]
            if not isinstance(headers, (list, tuple)):
                raise TypeError(
                    'invalid headers: {}, should be list of strings'.format(headers))
            phrases = _read_conll(filename, encoding='utf-8', sep='\t', indexes=indexes, dropna=True)

            sentences = []
            df = pd.DataFrame(columns=["input_text", "target_text"])

            for phrase in tqdm(phrases, total=len(phrases)):

                idx, phrase = phrase
                tokens, entity_tags, link_tags = phrase[0], phrase[1], phrase[-3]

                MAX_LENGTH = 50
                if len(tokens) > MAX_LENGTH:
                    def divide_chunks(l, n):
                        for i in range(0, len(l), n):
                            yield l[i:i + n]

                    sublists_of_tokens = list(divide_chunks(tokens, MAX_LENGTH))
                    sublists_of_entity_tags = list(divide_chunks(entity_tags, MAX_LENGTH))
                    sublists_of_link_tags = list(divide_chunks(link_tags, MAX_LENGTH))
                    for sublist_of_tokens, sublist_of_entity_tags, sublist_of_link_tags \
                            in zip(sublists_of_tokens, sublists_of_entity_tags, sublists_of_link_tags):
                        entities = get_entities(sublist_of_tokens, sublist_of_entity_tags, sublist_of_link_tags)
                        item = get_item(sublist_of_tokens, sublist_of_entity_tags, sublist_of_link_tags)
                        sentences.append(item)
                else:
                    entities = get_entities(tokens, entity_tags, link_tags)
                    item = get_item(tokens, entity_tags, link_tags)
                    sentences.append(item)

            df = pd.DataFrame.from_dict(sentences)
            df.colums = ["input_text", "target_text"]
            print('Saved to {}'.format(filename.replace('.tsv', '_bart.csv')))
            df.to_csv(filename.replace('.tsv', '_bart.csv'))
