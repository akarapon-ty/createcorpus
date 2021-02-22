#word cleaning
import re,os
from deepcut import tokenize
from pythainlp.util import isthai
import nltk

from pythainlp.spell import NorvigSpellChecker
from pythainlp.corpus import ttc

from spellchecker import SpellChecker

import tensorflow as tf

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# tf.get_logger().setLevel('ERROR')
# physical_devices = tf.config.list_physical_devices('GPU')
# try:
#   # Disable all GPUS
#   tf.config.set_visible_devices([], 'GPU')
#   visible_devices = tf.config.get_visible_devices()
#   for device in visible_devices:
#     assert device.device_type != 'GPU'
# except:
#   # Invalid device or cannot modify virtual devices once initialized.
#   pass


SPELL_ENG = SpellChecker()
# corpus = ttc.word_freqs()
# SPELL_THAI = NorvigSpellChecker(custom_dict=corpus)
SPELL_THAI = NorvigSpellChecker()

def load_corpus_thai():
    def readfile():
        path = "./corpus_thai_spellcheck_max"
        with open(path, "r", encoding="utf8") as f:
            return f.read().splitlines()

    corpus_thai = readfile()
    return corpus_thai

CORPUS_THAI = load_corpus_thai()

def removeSpaceLatin(word):
    removeSpace = re.sub(r'\s', '', word)
    return re.sub(r'[^\u0E00-\u0E7Fa-zA-Z0-9/.]', '', removeSpace)

def deepcut(line):
    list_word = tokenize(line)
    return list_word

def to_lower_case(s):
    if s[0].isupper():
        return s
    return str(s).lower()

def cleanWord(sentence):
    # Deep Cut process
    tokens = deepcut(sentence)
    # clean token
    tokens = list(map(removeSpaceLatin, tokens))
    tokens = list(map(removeSpaceLatin, tokens))
    while("" in tokens): 
        tokens.remove("") 
    tokens = list(map(to_lower_case, tokens))
    ###append new line or join to sperate sentence
    tokens.append('\n')
    return tokens

def spellCheckAuto(word):
    global SPELL_ENG
    global SPELL_THAI
    global CORPUS_THAI
    spellEngPrivate = SPELL_ENG
    spellThaiPrivate = SPELL_THAI
    corpusThai = CORPUS_THAI.copy()
    exception = ['ฯ', 'ๆ', '\n']

    if(word[0].isupper()):
        return word + " "

    if (word in exception):
        return word + " "

    isThai = isthai(word)
    if isThai:
        if(word in corpusThai):
            return word + " "
        return spellThaiPrivate.correct(word) + " "
    return spellEngPrivate.correction(word) + " "


def initNLTKCorpus():
    print("\nINIT NLTK Corpus")
    nltk.download('all')
    print()
