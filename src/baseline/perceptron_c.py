# Wrapper around simple perceptron/averaged perceptron C-library.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0

# Relies on C-code in libpercetron.so built from percetron.c through ctypes.
# Author: Mans Hulden

from ctypes import *

perceptron = cdll.LoadLibrary('./libperceptron.so')

perceptron_perceptron_init = perceptron.perceptron_init
perceptron_perceptron_init.restype = c_void_p
perceptron_examples_add = perceptron.examples_add
perceptron_examples_add.restype = None
perceptron_devexamples_add = perceptron.devexamples_add
perceptron_devexamples_add.restype = None
perceptron_perceptron_train = perceptron.perceptron_train
perceptron_perceptron_train.restype = None
perceptron_perceptron_classify_int = perceptron.perceptron_classify_int
perceptron_perceptron_classify_int.restype = c_int
perceptron_perceptron_classify_double = perceptron.perceptron_classify_double
perceptron_perceptron_classify_double.restype = c_int
perceptron_perceptron_decision_function_double = perceptron.perceptron_decision_function_double
perceptron_perceptron_decision_function_double.restype = POINTER(c_double)
perceptron_perceptron_decision_function_int = perceptron.perceptron_decision_function_int
perceptron_perceptron_decision_function_int.restype = POINTER(c_int)
perceptron_perceptron_free_wrapper = perceptron.perceptron_free_wrapper
perceptron_perceptron_destroy = perceptron.perceptron_destroy

class Perceptron:

    def __init__(self, max_iter = 20, averaged = False, shuffle = True, random_seed = False, tune_on_averaged = False, verbose = False):
        self.max_iter = max_iter
        self.averaged = averaged
        self.shuffle = shuffle
        self.random_seed = random_seed
        self.verbose = verbose
        self.tune_on_averaged = tune_on_averaged
        self.perceptronhandle = None

    def __del__(self):
        if self.perceptronhandle:
            perceptron_perceptron_destroy(c_void_p(self.perceptronhandle))
            
    def fit(self, features, classes, devfeatures = [], devclasses = []):
        # Map features to integers starting from 0
        self.num_examples = len(features)
        self.num_devexamples = len(devfeatures)
        fset = sorted(list(set([f for g in features + devfeatures for f in g])))
        self.inttofeat = dict(zip(xrange(len(fset)), fset))
        self.feattoint = dict(zip(fset, xrange(len(fset))))
        self.features = [[self.feattoint[f] for f in g] for g in features]
        self.num_features = len(fset)
        self.devfeatures = [[self.feattoint[f] for f in g] for g in devfeatures]
        
        # Map classes to integers starting from 0        
        cset = sorted(list(set([c for c in classes + devclasses])))
        self.inttoclass = dict(zip(xrange(len(cset)), cset))
        self.classtoint = dict(zip(cset, xrange(len(cset))))
        self.classes = [self.classtoint[f] for f in classes]
        self.devclasses = [self.classtoint[f] for f in devclasses]
        self.num_classes = len(cset)

        self.perceptronhandle = perceptron_perceptron_init(c_int(self.max_iter), c_int(self.num_examples), c_int(self.num_devexamples), c_int(self.num_features), c_int(self.num_classes), c_int(self.averaged), c_int(self.shuffle), c_int(self.random_seed), c_int(self.tune_on_averaged), c_int(self.verbose))

        for index, example_fs in enumerate(self.features):
            f = (c_int * len(example_fs))(*example_fs)
            perceptron_examples_add(c_void_p(self.perceptronhandle), f, c_int(len(example_fs)), c_int(self.classes[index]))

        for index, example_fs in enumerate(self.devfeatures):
            f = (c_int * len(example_fs))(*example_fs)
            perceptron_devexamples_add(c_void_p(self.perceptronhandle), f, c_int(len(example_fs)), c_int(self.devclasses[index]))
            
        perceptron_perceptron_train(c_void_p(self.perceptronhandle))

    def decision_function(self, features):
        test_fs = [self.feattoint[f] for f in features if f in self.feattoint]
        f = (c_int * len(test_fs))(*test_fs)
        if self.averaged:
            classweights = perceptron_perceptron_decision_function_double(c_void_p(self.perceptronhandle), f, c_int(len(test_fs)))
        else:
            classweights =  perceptron_perceptron_decision_function_int(c_void_p(self.perceptronhandle), f, c_int(len(test_fs)))
        c = [(self.inttoclass[i], classweights[i]) for i in xrange(self.num_classes)]
        perceptron_perceptron_free_wrapper(classweights)
        return c
        
    def predict(self, features):
        test_fs = [self.feattoint[f] for f in features if f in self.feattoint]
        f = (c_int * len(test_fs))(*test_fs)
        if self.averaged:
            correctclass = perceptron_perceptron_classify_double(c_void_p(self.perceptronhandle), f, c_int(len(test_fs)), c_int(0), c_int(0))
        else:
            correctclass = perceptron_perceptron_classify_int(c_void_p(self.perceptronhandle), f, c_int(len(test_fs)))
        return self.inttoclass[correctclass]

if __name__ == "__main__":
    P = Perceptron(shuffle = True, averaged = True, verbose = True)
    # 4 training examples, no dev examples (can use any data type for features)
    # We simply list the 'hot' features for each example 
    features = [['w','x','y','z'], ['u','w','x'],[232,'w'],[232,'x','y','z']]
    # The corresponding classes
    classes = ['CLASS_A','CLASS_A','CLASS_B','CLASS_A']
    # Train
    P.fit(features, classes)
    # Show probabilities of the classes for an instance
    print P.decision_function([232,'w','z']) # Print weights for classes
    # Show how the classes correspond to indices
    print P.classtoint
    # Show the best class for example
    print P.predict([232,'w','z'])
