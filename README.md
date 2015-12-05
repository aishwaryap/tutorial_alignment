# tutorial_alignment
This project uses latent models to align different tutorials for the same task. The model should also discriminate between tutorials of different tasks.

HMM code has been obtained from https://github.com/guyz/HMM

To run the code:

In one terminal - 
    cd stanford-corenlp-python
    python corenlp.py
    
Wait for Stanford CoreNLP to start up. Then in another terminal, 
    cd hmm/recipe/bigram
    python TestRecipe.py
    
Other language model based HMMs can be run in the same manner from folders hmm/recipe/background, hmm/recipe/verb_noun_unigram and hmm/recipe/verb_noun_tuple.
