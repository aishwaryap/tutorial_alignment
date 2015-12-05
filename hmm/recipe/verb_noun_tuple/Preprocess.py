import json, sys, re, time
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
            if len(text) > 1000 :
                paras = text.split('\n')
                idx = 0
                while idx < len(paras) :
                    part = ''
                    while len(part) < 500 and idx < len(paras) :
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
                    word_pos = zip(sentence, pos)
                    lemmatized_sentence = [(lemmatize(lmtzr, word, pos), pos) for (word, pos) in word_pos] 
                    verbs = [word for (word, pos) in lemmatized_sentence if pos in verb_pos_tags]
                    deps = sentence_object[u'dependencies']
                    for verb in verbs :
                        verb_with_nouns = (verb, )
                        obj_noun = [str(word2) for [dep_type, word1, word2] in deps if str(word1) == verb and dep_type == 'dobj']
                        other_relevant_noun = [str(word2) for [dep_type, word1, word2] in deps if str(word1) == verb and dep_type == 'nmod']
                        if len(obj_noun) > 0 :
                            if len(other_relevant_noun) > 0 :
                                verb_with_nouns = (verb, obj_noun[0], other_relevant_noun[0])
                            else :
                                verb_with_nouns = (verb, obj_noun[0])
                        else :
                            if len(other_relevant_noun) > 0 :
                                verb_with_nouns = (verb, other_relevant_noun[0])
                        #print 'verb_with_nouns = ', verb_with_nouns
                        phrases.append(verb_with_nouns)    
            if len(phrases) > 0 :
                recipes.append(phrases)
            orig_recipe_texts.append(orig_recipe_text)
            filenames.append(filename)
    if len(recipes) == 0 :
        recipes = ['-UNK-']
    print recipes
    return (recipes, filenames, orig_recipe_texts)        

def preprocess_texts(recipes_texts, ground_truth_sequences) :
    nlp = StanfordNLP()
    lmtzr = WordNetLemmatizer()
    recipes = list()
    orig_recipe_texts = list()
    extended_ground_truth_sequences = list()
    for (i, text) in enumerate(recipes_texts) :
        ground_truth = ground_truth_sequences[i]
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
        extended_ground_truth_sequence = list()
        for part in parts :
            lines = part.split('\n')
            text = ' '.join(lines)
            text = remove_non_ascii_chars(text)
            parse = nlp.parse(text.strip())
            state_idx = 0
            for sentence_object in parse[u'sentences'] :
                #tree = Tree.parse(sentence_object[u'parsetree'])
                #pprint(tree)
                sentence = [str(word[0]).lower() for word in sentence_object[u'words']]
                cur_orig_text = ' '.join([str(word[0]) for word in sentence_object[u'words']])
                orig_recipe_text.append(cur_orig_text)
                pos = [str(word[1][u'PartOfSpeech']) for word in sentence_object[u'words']]
                word_pos = zip(sentence, pos)
                lemmatized_sentence = [(lemmatize(lmtzr, word, pos), pos) for (word, pos) in word_pos] 
                verbs = [word for (word, pos) in lemmatized_sentence if pos in verb_pos_tags]
                deps = sentence_object[u'dependencies']
                for verb in verbs :
                    verb_with_nouns = (verb, )
                    obj_noun = [str(word2) for [dep_type, word1, word2] in deps if str(word1) == verb and dep_type == 'dobj']
                    other_relevant_noun = [str(word2) for [dep_type, word1, word2] in deps if str(word1) == verb and dep_type == 'nmod']
                    if len(obj_noun) > 0 :
                        if len(other_relevant_noun) > 0 :
                            verb_with_nouns = (verb, obj_noun[0], other_relevant_noun[0])
                        else :
                            verb_with_nouns = (verb, obj_noun[0])
                    else :
                        if len(other_relevant_noun) > 0 :
                            verb_with_nouns = (verb, other_relevant_noun[0])
                    #print 'verb_with_nouns = ', verb_with_nouns
                    phrases.append(verb_with_nouns)    
                    extended_ground_truth_sequence.append(ground_truth[state_idx])
                if state_idx < len(ground_truth) - 1 :
                    state_idx += 1
        if len(phrases) == 0 :
            phrases = ['-UNK-']
        recipes.append(phrases)
        orig_recipe_texts.append(orig_recipe_text)
        extended_ground_truth_sequences.append(extended_ground_truth_sequence)
    return (recipes, orig_recipe_texts, extended_ground_truth_sequences)        
