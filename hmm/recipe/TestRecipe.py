import numpy, json, sys
from pprint import pprint
from nltk.tree import Tree

from RecipeHMM import RecipeHMM

sys.path.insert(0, '../../stanford-corenlp-python')
from jsonrpc import ServerProxy, JsonRpc20, TransportTcpIp

class StanfordNLP:
    def __init__(self):
        self.server = ServerProxy(JsonRpc20(),
                                  TransportTcpIp(addr=("127.0.0.1", 8080)))
    
    def parse(self, text):
        return json.loads(self.server.parse(text))  

def test():
    nlp = StanfordNLP()
    text = 'Hi, my name is Aishwarya. I am testing whether this works.'
    parse = nlp.parse(text)
    print parse[u'sentences'][0].keys()
    print [str(word[0]) for word in parse[u'sentences'][0][u'words']]

    #observations = [[['a', 'cat'],['hello']],[['it', 'is', 'warm'],['a', 'dog'],['hello']]]
    
    #atmp = numpy.random.random_sample((2, 2))
    #row_sums = atmp.sum(axis=1)
    #a = atmp / row_sums[:, numpy.newaxis]    

    #pitmp = numpy.random.random_sample((2))
    #pi = pitmp / sum(pitmp)
    
    #hmm = RecipeHMM(2, pi, a)
    #hmm.train(observations, 100)
    #print "Pi", hmm.pi
    #print "A", hmm.A
    #print "uni", hmm.uni
    #print "bi", hmm.bi

if __name__ == '__main__' :
    test()
