import numpy, sys, copy, re
import editdistance
from os import listdir
from os.path import isfile, join
from itertools import permutations

from Preprocess import *
#from Baseline import *

train_recipes_dirs = ['../../../train']

#test_recipes_dirs = ['../../../test/pbnj', '../../../test/chocolate_cake']
#test_recipes_dirs = ['../../../groundTruth/pbnj']
test_recipes_dirs = ['../../../groundTruth/chocolate_cake']

def map_states(old_states, new_states, seq) :
    new_seq = copy.deepcopy(seq)
    temp_new_states = list(new_states)
    temp_new_states = [chr(int(i) + ord('a')) for i in temp_new_states]
    temp_new_states = ''.join(temp_new_states)
    for (idx, state) in enumerate(old_states) :
        new_seq = re.sub(state, temp_new_states[idx], new_seq)
    for (idx, state) in enumerate(temp_new_states) :
        new_seq = re.sub(state, new_states[idx], new_seq)
    return new_seq

def get_least_edit_distance(seq_list_1, seq_list_2, n) :
    min_sum_edit_distance = sys.maxint
    min_mapping = None
    states = ''.join([str(i) for i in range(n)])
    perms = [''.join(p) for p in permutations(states)]
    for perm in perms :
        total_dist = 0
        for (idx, seq1) in enumerate(seq_list_1) :
            seq2 = seq_list_2[idx]
            new_seq = map_states(states, perm, seq1)
            total_dist += int(editdistance.eval(new_seq, seq2))
        if total_dist < min_sum_edit_distance :
            min_sum_edit_distance = total_dist
            min_mapping = perm
    dist_list = list()
    for (idx, seq1) in enumerate(seq_list_1) :
        seq2 = seq_list_2[idx]
        new_seq = map_states(states, min_mapping, seq1)
        dist_list.append(int(editdistance.eval(new_seq, seq2)))
    return dist_list

def test_sequence_alignment() :
    (recipes, filenames, orig_recipe_texts) = preprocess(train_recipes_dirs)
    print 'Completed preprocessing'
    print 'Train files :', filenames
    n = 6
    
    atmp = numpy.random.random_sample((n, n))
    row_sums = atmp.sum(axis=1)
    a = atmp / row_sums[:, numpy.newaxis]    

    pitmp = numpy.random.random_sample((n))
    pi = pitmp / sum(pitmp)
    
    hmm = RecipeHMM(n, pi, a)
    hmm.train(recipes, 100)

    recipefiles = []
    recipe_texts = list()
    filenames = list()
    ground_truth_sequences = list()
    for recipes_dir in test_recipes_dirs :
        print 'recipes_dir = ', recipes_dir
        recipefiles = recipefiles + [ recipes_dir + '/' + f for f in listdir(recipes_dir) if isfile(join(recipes_dir,f)) ]
    for filename in recipefiles :
        if filename.endswith('.txt') :
            filenames.append(filename)
            f = open(filename)
            print 'Processing ', filename
            text = f.read()
            f.close()
            lines = text.strip().split('\n')
            text = list()
            sequence = list()
            for line in lines :
                [label, line_text] = line.split('|')
                text.append(line_text)
                sequence.append(label)
            text = '\n'.join(text)
            recipe_texts.append(text)
            ground_truth_sequences.append(sequence)
    
    (recipes, orig_recipe_texts) = preprocess_texts(recipe_texts)
    all_recipes_states = list()
    ground_truth_str_sequences = list()
    for (recipe, ground_truth) in zip(recipes, ground_truth_sequences) :
        states =  hmm.decode(recipe)
        states = ''.join([str(int(state)) for state in list(states)])
        ground_truth = ''.join([str(int(state)) for state in list(ground_truth)])
        all_recipes_states.append(states)
        ground_truth_str_sequences.append(ground_truth)
        # Note: The lengths of the two sequences need not be the same
        
    dist_list = get_least_edit_distance(all_recipes_states, ground_truth_str_sequences, n)
    
    for (recipe, filename, dist) in zip(recipes, filenames, dist_list) :
        print filename, ',', dist, ',', len(recipe) 
        
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
    #print map_states('01234', '56789', '12341')
    #print get_least_edit_distance('01234', '12340', 5)
    test_sequence_alignment()
    #test()
    #test_baseline()
