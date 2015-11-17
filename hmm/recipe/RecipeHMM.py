'''
This code is based on:
 - QSTK's HMM implementation - http://wiki.quantsoftware.org/
 - A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition, LR RABINER 1989 
'''

import numpy

class RecipeHMM :
    
    def __init__(self, n, pi=None, A=None, precision=numpy.double, verbose=False):
        self.n = n
        
        self.precision = precision
        self.verbose = verbose
        self._eta = self._eta1
        
        self.A = A
        self.pi = pi
        self.uni = dict()
        self.uni['-UNK-'] = 1.0
        self.bi = dict()
        self.bi['-UNK-'] = 1.0

    def _eta1(self,t,T):
        '''
        Governs how each sample in the time series should be weighed.
        This is the default case where each sample has the same weigh, 
        i.e: this is a 'normal' HMM.
        '''
        return 1.
    
    def calc_uni(self, j, w) :
        if w in uni[j] :
            return uni[j][w]
        else :
            return uni[j]['-UNK-']
    
    def calc_bi(self, j, w1, w2) :
        if (w1, w2) in bi[j] :
            return bi[j][(w1, w2)]
        else :
            return bi[j]['-UNK-']
    
    
    # o = [w_1, w_2 ... w_n]
    # b_j(o) = Pr()  
    def calc_b(self, j, o) :
        for i in xrange(len(o)) :
            b *= (self.calc_bi(j, o[i-1], o[i]) / self.calc_uni(j, o[i-1])
        return b    
            
    def forwardbackward(self, observations, cache=False):
        '''
        This should be computed separately for each sequence of observations
        
        Forward-Backward procedure is used to efficiently calculate the probability of the observation, given the model - P(O|model)
        alpha_t(x) = P(O1...Ot,qt=Sx|model) - The probability of state x and the observation up to time t, given the model.
        
        The returned value is the log of the probability, i.e: the log likehood model, give the observation - logL(model|O).
        
        In the discrete case, the value returned should be negative, since we are taking the log of actual (discrete)
        probabilities. In the continuous case, we are using PDFs which aren't normalized into actual probabilities,
        so the value could be positive.
        '''
        if (cache==False):
            print 'Using b values from only one observation. This should not happen!'
            self._mapB([observations])
        
        alpha = self._calcalpha(observations)
        # Pr(O/params) = \sum_i(\alpha_T(i))
        return numpy.log(sum(alpha[-1]))
    
    def _calcalpha(self,observations):
        '''
        This should be computed separately for each sequence of observations
        
        Calculates 'alpha' the forward variable.
    
        The alpha variable is a numpy array indexed by time, then state (TxN).
        alpha[t][i] = the probability of being in state 'i' after observing the 
        first t symbols.
        alpha[t][i] = \alpha_t(i)
        '''        
        
        alpha = numpy.zeros((len(observations),self.n),dtype=self.precision)
        
        # init stage - alpha_1(x) = pi(x)b_x(O1)
        for x in xrange(self.n):
            alpha[0][x] = self.pi[x] * self.calc_b(x, observations[0])
        
        # induction
        for t in xrange(1,len(observations)):
            for j in xrange(self.n):
                for i in xrange(self.n):
                    alpha[t][j] += alpha[t-1][i]*self.A[i][j]
                alpha[t][j] *= self.calc_b(j, observations[t])
                
        return alpha

    def _calcbeta(self,observations):
        '''
        This should be computed separately for each sequence of observations
        
        Calculates 'beta' the backward variable.
        
        The beta variable is a numpy array indexed by time, then state (TxN).
        beta[t][i] = the probability of being in state 'i' and then observing the
        symbols from t+1 to the end (T).
        '''        
        beta = numpy.zeros((len(observations),self.n),dtype=self.precision)
        
        # init stage
        for s in xrange(self.n):
            beta[len(observations)-1][s] = 1.
        
        # induction
        for t in xrange(len(observations)-2,-1,-1):
            for i in xrange(self.n):
                for j in xrange(self.n):
                    beta[t][i] += self.A[i][j] * self.calc_b(j, observations[t+1]) * beta[t+1][j]
                    
        return beta
    
    def decode(self, observations):
        '''
        This should be computed separately for each sequence of observations
        
        Find the best state sequence (path), given the model and an observation. i.e: max(P(Q|O,model)).
        
        This method is usually used to predict the next state after training. 
        '''        
        # use Viterbi's algorithm. It is possible to add additional algorithms in the future.
        return self._viterbi(observations)
    
    # TODO : This hasn't been checked
    #def _viterbi(self, observations):
        #'''
        #This should be computed separately for each sequence of observations
        
        #Find the best state sequence (path) using viterbi algorithm - a method of dynamic programming,
        #very similar to the forward-backward algorithm, with the added step of maximization and eventual
        #backtracing.
        
        #delta[t][i] = max(P[q1..qt=i,O1...Ot|model] - the path ending in Si and until time t,
        #that generates the highest probability.
        
        #psi[t][i] = argmax(delta[t-1][i]*aij) - the index of the maximizing state in time (t-1), 
        #i.e: the previous state.
        #'''
        ## similar to the forward-backward algorithm, we need to make sure that we're using fresh data for the given observations.
        ## The following funcion call will not work as now _mapB uses all observations
        #self._mapB(observations)
        
        #delta = numpy.zeros((len(observations),self.n),dtype=self.precision)
        #psi = numpy.zeros((len(observations),self.n),dtype=self.precision)
        
        ## init
        #for x in xrange(self.n):
            #delta[0][x] = self.pi[x] * self.calc_b(x, observations[0])
            #psi[0][x] = 0
        
        ## induction
        #for t in xrange(1,len(observations)):
            #for j in xrange(self.n):
                #for i in xrange(self.n):
                    #if (delta[t][j] < delta[t-1][i]*self.A[i][j]):
                        #delta[t][j] = delta[t-1][i]*self.A[i][j]
                        #psi[t][j] = i
                #delta[t][j] *= self.calc_b(j, observations[t])
        
        ## termination: find the maximum probability for the entire sequence (=highest prob path)
        #p_max = 0 # max value in time T (max)
        #path = numpy.zeros((len(observations)),dtype=self.precision)
        #for i in xrange(self.n):
            #if (p_max < delta[len(observations)-1][i]):
                #p_max = delta[len(observations)-1][i]
                #path[len(observations)-1] = i
        
        ## path backtracing
##        path = numpy.zeros((len(observations)),dtype=self.precision) ### 2012-11-17 - BUG FIX: wrong reinitialization destroyed the last state in the path
        #for i in xrange(1, len(observations)):
            #path[len(observations)-i-1] = psi[len(observations)-i][ path[len(observations)-i] ]
        #return path
     
    def _calcxi(self,observations,alpha=None,beta=None):
        '''
        This should be computed separately for each sequence of observations
        
        Calculates 'xi', a joint probability from the 'alpha' and 'beta' variables.
        
        The xi variable is a numpy array indexed by time, state, and state (TxNxN).
        xi[t][i][j] = the probability of being in state 'i' at time 't', and 'j' at
        time 't+1' given the entire observation sequence.
        
        This refers to the weird Greek letter xi
        xi_t(i,j) = Pr(q_t = S_i, q_{t+1} = S_j/ O, lambda)
        '''        
        if alpha is None:
            alpha = self._calcalpha(observations)
        if beta is None:
            beta = self._calcbeta(observations)
        xi = numpy.zeros((len(observations),self.n,self.n),dtype=self.precision)
        
        for t in xrange(len(observations)-1):
            denom = 0.0
            for i in xrange(self.n):
                for j in xrange(self.n):
                    thing = 1.0
                    thing *= alpha[t][i]
                    thing *= self.A[i][j]
                    thing *= self.calc_b(j, observations[t+1])
                    thing *= beta[t+1][j]
                    denom += thing
            for i in xrange(self.n):
                for j in xrange(self.n):
                    numer = 1.0
                    numer *= alpha[t][i]
                    numer *= self.A[i][j]
                    numer *= self.calc_b(j, observations[t+1])
                    numer *= beta[t+1][j]
                    xi[t][i][j] = numer/denom
                    
        return xi

    def _calcgamma(self,xi,seqlen):
        '''
        This should be computed separately for each sequence of observations
        
        Calculates 'gamma' from xi.
        
        Gamma is a (TxN) numpy array, where gamma[t][i] = the probability of being
        in state 'i' at time 't' given the full observation sequence.
        '''        
        gamma = numpy.zeros((seqlen,self.n),dtype=self.precision)
        
        for t in xrange(seqlen):
            for i in xrange(self.n):
                gamma[t][i] = sum(xi[t][i])
        
        return gamma
    
    def train(self, observations, iterations=1,epsilon=0.0001,thres=-0.001):
        '''
        observations is a list of observation sequences
        
        Updates the HMMs parameters given a new set of observed sequences.
        
        observations can either be a single (1D) array of observed symbols, or when using
        a continuous HMM, a 2D array (matrix), where each row denotes a multivariate
        time sample (multiple features).
        
        Training is repeated 'iterations' times, or until log likelihood of the model
        increases by less than 'epsilon'.
        
        'thres' denotes the algorithms sensitivity to the log likelihood decreasing
        from one iteration to the other.
        '''        
        self._mapB(observations)
        
        for i in xrange(iterations):
            prob_old, prob_new = self.trainiter(observations)

            if (self.verbose):      
                print "iter: ", i, ", L(model|O) =", prob_old, ", L(model_new|O) =", prob_new, ", converging =", ( prob_new-prob_old > thres )
                
            if ( abs(prob_new-prob_old) < epsilon ):
                # converged
                break
    
    # TODO: Check if something was added to new_model            
    def _updatemodel(self,new_model):
        '''
        Replaces the current model parameters with the new ones.
        '''
        self.pi = new_model['pi']
        self.A = new_model['A']
                
    def trainiter(self,observations):
        '''
        observations is a list of observation sequences
        
        A single iteration of an EM algorithm, which given the current HMM,
        computes new model parameters and internally replaces the old model
        with the new one.
        
        Returns the log likelihood of the old model (before the update),
        and the one for the new model.
        '''        
        # call the EM algorithm
        new_model = self._baumwelch(observations)
        
        # calculate the log likelihood of the previous model
        prob_old = 0.0
        for observation in observations :
            prob_old += self.forwardbackward(observation, cache=True)
        
        # update the model with the new estimation
        self._updatemodel(new_model)
        
        # Recalculate b
        self._mapB(observations)
        
        # calculate the log likelihood of the new model. Cache set to false in order to recompute probabilities of the observations give the model.
        prob_new = 0.0
        for observation in observations :
            prob_new += self.forwardbackward(observation, cache=True)
        
        return prob_old, prob_new
    
    def _reestimateA(self,observations,stats):
        '''
        observations is a list of observation sequences
        
        Reestimation of the transition matrix (part of the 'M' step of Baum-Welch).
        Computes A_new = expected_transitions(i->j)/expected_transitions(i)
        
        Returns A_new, the modified transition matrix. 
        '''
        A_new = numpy.zeros((self.n,self.n),dtype=self.precision)
        
        for k in xrange(len(observations)) :
            P_k = sum(stats['alpha'][k][-1])
            for i in xrange(self.n):
                for j in xrange(self.n):        
                    term = 0.0
                    for t in xrange(len(observations[k])-1): 
                        term += stats['alpha'][k][t][i] * self.A[i][j] * self.calc_b(j, observations[k][t+1]) * stats['beta'][k][t+1][j]
                    A_new[i][j] += term / P_k
                    
        return A_new
    
    def _calcstats(self,observations):
        '''
        observations is a list of observation sequences
        
        Calculates required statistics of the current model, as part
        of the Baum-Welch 'E' step.
        
        Deriving classes should override (extend) this method to include
        any additional computations their model requires.
        
        Returns 'stat's, a dictionary containing required statistics.
        '''
        stats = dict()
        stats['alpha'] = list()
        stats['beta'] = list()
        stats['xi'] = list()
        stats['gamma'] = list()
            
        for (k, observation) in enumerate(observations) :
            stats['alpha'].append(self._calcalpha(observation))
            stats['beta'].append(self._calcbeta(observation))
            stats['xi'].append(self._calcxi(observation, stats['alpha'][k], stats['beta'][k]))
            stats['gamma'].append(self._calcgamma(stats['xi'][k], len(observation)))
        
        return stats
    
    def _reestimate(self,stats,observations):
        '''
        observations is a list of observation sequences
        
        Performs the 'M' step of the Baum-Welch algorithm.
        
        Deriving classes should override (extend) this method to include
        any additional computations their model requires.
        
        Returns 'new_model', a dictionary containing the new maximized
        model's parameters.
        '''        
        new_model = {}
        
        # new init vector is set to the frequency of being in each step at t=0 
        new_model['pi'] = stats['gamma'][0][0]
        new_model['A'] = self._reestimateA(observations, stats)
        
        return new_model
    
    def _baumwelch(self,observations):
        '''
        observations is a list of observation sequences
        
        An EM(expectation-modification) algorithm devised by Baum-Welch. Finds a local maximum
        that outputs the model that produces the highest probability, given a set of observations.
        
        Returns the new maximized model parameters
        '''        
        # E step - calculate statistics
        stats = self._calcstats(observations)
        
        # M step
        return self._reestimate(stats,observations)

    def _mapB(self, observations, stats):
        '''
        observations is a list of observation sequences
        '''
        self.uni = [dict()] * self.n
        self.bi = [dict()] * self.n
        for (k, observation) in enumerate(observations) :
            for (t, phrase) in enumerate(observation) :
                # Assuming observation is a list of unigrams
                unigrams = ['<s>'] + phrase + ['</s>']
                bigrams = zip(unigrams, unigrams[1:])
                for j in xrange(self.n) :
                    sum_uni = 0.0
                    sum_bi = 0.0
                    for unigram in unigrams :
                        if unigram in self.uni[j] :
                            self.uni[j][unigram] += stats['gamma'][k][t][j]
                        else :
                            self.uni[j][unigram] = stats['gamma'][k][t][j]
                        sum_uni += stats['gamma'][k][t][j]
                    # TODO: Recheck the correct way to init this
                    self.uni[j]['-UNK-'] = 1
                    sum_uni += self.uni[j]['-UNK-']
                    for bigram in bigrams :
                        if bigram in self.bi[j] :
                            self.bi[j][bigram] += stats['gamma'][k][t][j]
                        else :
                            self.bi[j][bigram] = stats['gamma'][k][t][j]
                        sum_bi += stats['gamma'][k][t][j]
                    self.bi[j]['-UNK-'] = 1
                    sum_bi += self.bi[j]['-UNK-']
                    
                    for unigram in self.uni[j] :
                        self.uni[j][unigram] /= sum_uni
                    for bigram in bi[j] :
                        self.bi[j][bigram] /= sum_bi
                
        
        
