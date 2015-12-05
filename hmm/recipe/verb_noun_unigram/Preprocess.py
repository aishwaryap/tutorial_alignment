import json, sys, re, time, copy
from pprint import pprint
from nltk.tree import Tree
from os import listdir
from os.path import isfile, join
from nltk.tree import Tree
from nltk.stem.wordnet import WordNetLemmatizer

from RecipeHMM import RecipeHMM

sys.path.insert(0, '../../../stanford-corenlp-python')
from jsonrpc import ServerProxy, JsonRpc20, TransportTcpIp, RPCTransportError

class StanfordNLP:
    def __init__(self):
        self.server = ServerProxy(JsonRpc20(),
                                  TransportTcpIp(addr=("127.0.0.1", 8080)))
    
    def parse(self, text):
        return json.loads(self.server.parse(text))  


verb_pos_tags = ['VBG', 'VBN', 'VBD', 'VBP', 'VBZ', 'VB']
adj_pos_tags = ['JJ', 'JJR', 'JJS']
noun_pos_tags = ['NN', 'NNS', 'NNP', 'NNPS']
adv_pos_tags = ['RB', 'RBR', 'RBS']

delete_list = ['-rrb-', '-lrb-', '-RRB-', '-LRB-']

def get_only_words(sentence) :
    regex = re.compile('.*[a-zA-Z0-9].*')
    only_words = [(token.lower(), pos) for (token, pos) in sentence if regex.match(token) and token not in delete_list]
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

def preprocess(recipes_dirs) :
    recipefiles = []
    for recipes_dir in recipes_dirs :
        print 'recipes_dir = ', recipes_dir
        recipefiles = recipefiles + [ recipes_dir + '/' + f for f in listdir(recipes_dir) if isfile(join(recipes_dir,f)) ]
    nlp = StanfordNLP()
    lmtzr = WordNetLemmatizer()
    recipes = list()
    filenames = list()
    orig_recipe_texts = list()
    for filename in recipefiles :
        if filename.endswith('.txt') :
            print 'Preprocessing ', filename
            f = open(filename)
            text = f.read()
            parts = list()
            if len(text) > 2000 :
                paras = text.split('\n')
                idx = 0
                while idx < len(paras) :
                    part = ''
                    while len(part) < 1000 and idx < len(paras) :
                        part = part + paras[idx] + '\n'
                        idx += 1
                    parts.append(part.strip())
            else :
                parts = [text]

            f.close()
            phrases = list()    
            orig_recipe_text = list()
            for part in parts :
                lines = part.split('\n')
                text = ' '.join(lines)
                text = remove_non_ascii_chars(text)
                parse = nlp.parse(text.strip())
                for sentence_object in parse[u'sentences'] :
                    #tree = Tree.parse(sentence_object[u'parsetree'])
                    #pprint(tree)
                    sentence = [str(word[0]).lower() for word in sentence_object[u'words']]
                    cur_orig_text = ' '.join([str(word[0]) for word in sentence_object[u'words']])
                    orig_recipe_text.append(cur_orig_text)
                    pos = [str(word[1][u'PartOfSpeech']) for word in sentence_object[u'words']]
                    pos_copy = copy.deepcopy(pos)
                    word_pos = zip(sentence, pos)
                    #print 'word_pos = ', word_pos
                    lemmatized_sentence_with_pos = [(lemmatize(lmtzr, word, pos), pos) for (word, pos) in word_pos] 
                    #print 'lemmatized_sentence = ', lemmatized_sentence
                    #print 'pos = ', pos
                    #lemmatized_sentence_with_pos = zip(lemmatized_sentence, pos_copy)
                    #print 'lemmatized_sentence_with_pos = ', lemmatized_sentence_with_pos
                    only_words = get_only_words(lemmatized_sentence_with_pos)
                    #print 'only_words = ',only_words
                    verbs_and_nouns = [word for (word, pos) in only_words if pos in verb_pos_tags or pos in noun_pos_tags]
                    #print'verbs_and_nouns = ',verbs_and_nouns 
                    phrases.append(verbs_and_nouns)
            recipes.append(phrases)
            orig_recipe_texts.append(orig_recipe_text)
            filenames.append(filename)
    return (recipes, filenames, orig_recipe_texts)        

def preprocess_texts(recipes_texts) :
    nlp = StanfordNLP()
    lmtzr = WordNetLemmatizer()
    recipes = list()
    orig_recipe_texts = list()
    for (i, text) in enumerate(recipes_texts) :
        print 'Preprocessing text file ', i 
        #print text
        parts = list()
        if len(text) > 2000 :
            paras = text.split('\n')
            idx = 0
            while idx < len(paras) :
                part = ''
                while len(part) < 1000 and idx < len(paras) :
                    part = part + paras[idx] + '\n'
                    idx += 1
                parts.append(part.strip())
        else :
            parts = [text]

        phrases = list()    
        orig_recipe_text = list()
        for part in parts :
            lines = part.split('\n')
            text = ' '.join(lines)
            text = remove_non_ascii_chars(text)
            parse = nlp.parse(text.strip())
            for sentence_object in parse[u'sentences'] :
                #tree = Tree.parse(sentence_object[u'parsetree'])
                #pprint(tree)
                sentence = [str(word[0]).lower() for word in sentence_object[u'words']]
                cur_orig_text = ' '.join([str(word[0]) for word in sentence_object[u'words']])
                orig_recipe_text.append(cur_orig_text)
                pos = [str(word[1][u'PartOfSpeech']) for word in sentence_object[u'words']]
                pos_copy = copy.deepcopy(pos)
                word_pos = zip(sentence, pos)
                #print 'word_pos = ', word_pos
                lemmatized_sentence_with_pos = [(lemmatize(lmtzr, word, pos), pos) for (word, pos) in word_pos] 
                #print 'lemmatized_sentence = ', lemmatized_sentence
                #print 'pos = ', pos
                #lemmatized_sentence_with_pos = zip(lemmatized_sentence, pos_copy)
                #print 'lemmatized_sentence_with_pos = ', lemmatized_sentence_with_pos
                only_words = get_only_words(lemmatized_sentence_with_pos)
                #print 'only_words = ',only_words
                verbs_and_nouns = [word for (word, pos) in only_words if pos in verb_pos_tags or pos in noun_pos_tags]
                #print'verbs_and_nouns = ',verbs_and_nouns 
                phrases.append(verbs_and_nouns)
        recipes.append(phrases)
        orig_recipe_texts.append(orig_recipe_text)
    return (recipes, orig_recipe_texts)        
