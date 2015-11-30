import numpy, sys
from os import listdir
from os.path import isfile, join

from Preprocess import *
#from Baseline import *

train_recipes_dirs = ['../../../train']

#test_recipes_dirs = ['../../../test/pbnj', '../../../test/chocolate_cake']
test_recipes_dirs = ['../../../groundTruth/pbnj']

def test_sequence_alignment() :
    (recipes, filenames, orig_recipe_texts) = preprocess(train_recipes_dirs)
    print 'Completed preprocessing'
    print 'Train files :', filenames
    n = 9
    
    atmp = numpy.random.random_sample((n, n))
    row_sums = atmp.sum(axis=1)
    a = atmp / row_sums[:, numpy.newaxis]    

    pitmp = numpy.random.random_sample((n))
    pi = pitmp / sum(pitmp)
    
    hmm = RecipeHMM(n, pi, a)
    hmm.train(recipes, 100)

    recipefiles = []
    recipe_texts = list()
    ground_truth_sequences = list()
    for recipes_dir in test_recipes_dirs :
        print 'recipes_dir = ', recipes_dir
        recipefiles = recipefiles + [ recipes_dir + '/' + f for f in listdir(recipes_dir) if isfile(join(recipes_dir,f)) ]
    for filename in recipefiles :
        if filename.endswith('.txt') :
            f = open(filename)
            text = f.read()
            f.close()
            lines = text.split('\n')
            text = list()
            sequence = list()
            for line in lines :
                [label, line_text] = line.split('|')
                text.append(line_text)
                sequence.append(label)
            text = ' '.join(text)
            recipe_texts.append(text)
            ground_truth_sequences.append(sequence)
    
    (recipes, orig_recipe_texts) = preprocess_texts(recipes_texts)
    for (recipe, orig_text, ground_truth) in zip(recipes, orig_recipe_texts, ground_truth_sequences) :
        states =  hmm.decode(recipe)
        lines = orig_text
        output = zip(states, lines)
        for (state, line) in output :
            print state, ' : ', line
        print '-------------------------------------------------'    
    
    #print "uni", hmm.uni
    #print "bi", hmm.bi


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
    n = 9
    
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
    test_sequence_alignment()
    #test()
    #test_baseline()
