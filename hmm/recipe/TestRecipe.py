import numpy
from os import listdir
from os.path import isfile, join

from Preprocess import *

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

train_recipes_dirs = ['../../data/pbnj/recipes']
train_ingredients_dir = ['../../data/pbnj/ingredients']

test_recipes_dirs = ['../../data/pbnj/recipes', '../../data/chocolate_cake/recipes']
test_ingredients_dir = ['../../data/pbnj/ingredients', '../../data/chocolate_cake/ingredients']

def test():
    (recipes, filenames) = preprocess(train_recipes_dirs)
    print 'Completed preprocessing'
    print 'Train files :', filenames
    #recipes = recipes + recipes
    #temp = list()
    #temp.append(recipes[1])
    #temp.append(recipes[0])
    #recipes = temp
    n = len(recipes[0])
    
    atmp = numpy.random.random_sample((n, n))
    row_sums = atmp.sum(axis=1)
    a = atmp / row_sums[:, numpy.newaxis]    

    pitmp = numpy.random.random_sample((n))
    pi = pitmp / sum(pitmp)
    
    hmm = RecipeHMM(n, pi, a)
    hmm.train(recipes, 10)
    #print "Pi", hmm.pi
    #print "A", hmm.A

    (recipes, filenames) = preprocess(test_recipes_dirs)
    filenames_with_recipes = zip(filenames, recipes)
    filenames_with_recipes.sort()
    for (filename, recipe) in filenames_with_recipes :
        print filename, ':',  hmm.forwardbackward(recipe, cache=True)    

    
    
    #print "uni", hmm.uni
    #print "bi", hmm.bi

if __name__ == '__main__' :
    test()
