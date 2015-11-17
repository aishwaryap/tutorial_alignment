import numpy
from RecipeHMM import RecipeHMM

def test():
    observations = [[['a', 'cat'],['hello']],[['it', 'is', 'warm'],['a', 'dog'],['hello']]]
    
    atmp = numpy.random.random_sample((2, 2))
    row_sums = atmp.sum(axis=1)
    a = atmp / row_sums[:, numpy.newaxis]    

    pitmp = numpy.random.random_sample((2))
    pi = pitmp / sum(pitmp)
    
    hmm = RecipeHMM(2, pi, a)
    hmm.train(observations, 100)
    print "Pi", hmm.pi
    print "A", hmm.A
    print "uni", hmm.uni
    print "bi", hmm.bi

if __name__ == '__main__' :
    test()
