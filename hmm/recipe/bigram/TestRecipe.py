import numpy, sys
from os import listdir
from os.path import isfile, join

from Preprocess import *
#from Baseline import *

def basic_test() :
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

train_recipes_dirs = ['../../../train']
#train_recipes_dirs = ['../../trial']

test_recipes_dirs = ['../../../test/pbnj', '../../../test/chocolate_cake']
#test_recipes_dirs = ['../../test']

def test():
    (recipes, filenames, orig_recipe_texts) = preprocess(train_recipes_dirs)
    print 'Completed preprocessing'
    print 'Train files :', filenames
    #recipes = recipes + recipes
    #temp = list()
    #temp.append(recipes[1])
    #temp.append(recipes[0])
    #recipes = temp
    #n = len(recipes[0])
    n = 4
    
    atmp = numpy.random.random_sample((n, n))
    row_sums = atmp.sum(axis=1)
    a = atmp / row_sums[:, numpy.newaxis]    

    pitmp = numpy.random.random_sample((n))
    pi = pitmp / sum(pitmp)
    
    hmm = RecipeHMM(n, pi, a)
    hmm.train(recipes, 100)

    (recipes, filenames, orig_recipe_texts) = preprocess(test_recipes_dirs)
    filenames_with_recipes = zip(filenames, recipes)
    filenames_with_recipes.sort()
    for (filename, recipe) in filenames_with_recipes :
        print filename, ':',  hmm.forwardbackward(recipe, cache=True)    
    
    #for (recipe, orig_text) in zip(recipes, orig_recipe_texts) :
        #states =  hmm.decode(recipe)
        #lines = orig_text
        #output = zip(states, lines)
        #for (state, line) in output :
            #print state, ' : ', line
        #print '-------------------------------------------------'    
    
    #print "uni", hmm.uni
    #print "bi", hmm.bi

def test_baseline() :
    baseline(test_recipes_dirs)

if __name__ == '__main__' :
    test()
    #test_baseline()
