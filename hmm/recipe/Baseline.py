import json, sys, re, time
from pprint import pprint
from nltk.tree import Tree
from os import listdir
from os.path import isfile, join
from nltk.tree import Tree
from nltk.stem.wordnet import WordNetLemmatizer

from RecipeHMM import RecipeHMM
from Preprocess import *

sys.path.insert(0, '../../stanford-corenlp-python')
from jsonrpc import ServerProxy, JsonRpc20, TransportTcpIp, RPCTransportError

def get_verb_object_sequence(text, nlp) :
    lines = text.split('\n')
    text = ' '.join(lines)
    text = remove_non_ascii_chars(text)
    parse = nlp.parse(text.strip())
    for sentence_object in parse[u'sentences'] :
        tree = Tree(sentence_object[u'parsetree'])
        print '****************************************'
        print tree
        print '****************************************'
        print tree[0,0]
        print '****************************************'
        deps = sentence_object[u'dependencies']
        root_verb = [word2 for [dep_type, word1, word2] in deps if str(dep_type) == 'root' ][0]
        objs = [word2 for [dep_type, word1, word2] in deps if str(word1) == root_verb]
        print sentence_object[u'text']
        print root_verb
        print objs
        print '-----------------------------------'
        verbs = [(str(word[0]), word[1][u'PartOfSpeech'])  for word in sentence_object[u'words'] if word[1][u'PartOfSpeech'] in verb_pos_tags]
        #print verbs
    return []

def baseline(recipes_dirs) :
    recipefiles = []
    for recipes_dir in recipes_dirs :
        print 'recipes_dir = ', recipes_dir
        recipefiles = recipefiles + [ recipes_dir + '/' + f for f in listdir(recipes_dir) if isfile(join(recipes_dir,f)) ]
    nlp = StanfordNLP()
    for filename in recipefiles :
        sequence = []
        if filename.endswith('.txt') :
            print 'On ', filename
            f = open(filename)
            text = f.read()
            parts = list()
            if len(text) > 3000 :
                paras = text.split('\n')
                idx = 0
                while idx < len(paras) :
                    part = ''
                    while len(part) < 1500 and idx < len(paras) :
                        part = part + paras[idx] + '\n'
                        idx += 1
                    parts.append(part.strip())
            else :
                parts = [text]
            f.close()
            sequence = sequence + get_verb_object_sequence(text, nlp)
            
