import json, sys, re
from pprint import pprint
from nltk.tree import Tree
from os import listdir
from os.path import isfile, join
from nltk.tree import Tree
from nltk.stem.wordnet import WordNetLemmatizer

from RecipeHMM import RecipeHMM

sys.path.insert(0, '../../stanford-corenlp-python')
from jsonrpc import ServerProxy, JsonRpc20, TransportTcpIp

class StanfordNLP:
    def __init__(self):
        self.server = ServerProxy(JsonRpc20(),
                                  TransportTcpIp(addr=("127.0.0.1", 8080)))
    
    def parse(self, text):
        return json.loads(self.server.parse(text))  


# TODO: Remove non-ASCII characters from text before gving to stanford corenlp

verb_pos_tags = ['VBG', 'VBN', 'VBD', 'VBP', 'VBZ', 'VB']
adj_pos_tags = ['JJ', 'JJR', 'JJS']
noun_pos_tags = ['NN', 'NNS', 'NNP', 'NNPS']
adv_pos_tags = ['RB', 'RBR', 'RBS']

delete_list = ['-rrb-', '-lrb-']

def get_only_words(sentence) :
    regex = re.compile('.*[a-zA-Z0-9].*')
    only_words = [token.lower() for token in sentence if regex.match(token) and token not in delete_list]
    return only_words

def lemmatize(lmtzr, word, pos) :
    if pos in verb_pos_tags :
        lemma = str(lmtzr.lemmatize(word, 'v'))
    elif pos in noun_pos_tags :
        lemma = str(lmtzr.lemmatize(word, 'n'))
    elif pos in adj_pos_tags :
        lemma = str(lmtzr.lemmatize(word, 'a'))
    elif pos in adv_pos_tags:
        lemma = str(lmtzr.lemmatize(word, 'r'))
    else :
        lemma = str(lmtzr.lemmatize(word))
    return lemma

def remove_non_ascii_chars(text) :
    return ''.join([i if ord(i) < 128 else ' ' for i in text])

def preprocess(recipes_dir) :
    recipefiles = [ f for f in listdir(recipes_dir) if isfile(join(recipes_dir,f)) ]
    nlp = StanfordNLP()
    lmtzr = WordNetLemmatizer()
    recipes = list()
    filenames = list()
    for filename in recipefiles :
        if filename.endswith('.txt') :
            print 'Preprocessing ', filename
            f = open(recipes_dir + '/' + filename)
            text = f.read()
            lines = text.split('\n')
            text = ' '.join(lines)
            text = remove_non_ascii_chars(text)
            f.close()
            parse = nlp.parse(text.strip())
            phrases = list()
            for sentence_object in parse[u'sentences'] :
                #tree = Tree.parse(sentence_object[u'parsetree'])
                #pprint(tree)
                sentence = [str(word[0]).lower() for word in sentence_object[u'words']]
                pos = [str(word[1][u'PartOfSpeech']) for word in sentence_object[u'words']]
                word_pos = zip(sentence, pos)
                lemmatized_sentence = [lemmatize(lmtzr, word, pos) for (word, pos) in word_pos] 
                only_words = get_only_words(lemmatized_sentence)
                phrases.append(only_words)
            recipes.append(phrases)
            filenames.append(filename)
    return (recipes, filenames)        
