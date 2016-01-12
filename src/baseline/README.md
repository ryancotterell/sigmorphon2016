# Baseline for SIGMORPHON 2016

This is a simple baseline that solves tasks 1, 2, and 3 using only the labeled data provided.

To run the baseline on all the data sets and all tasks:

* Run make in this directory, which builds the C-dependencies that **baseline.py** depends on.

* Run **./runall.sh** which solves all the tasks in the data directory and stores the output as well as the evaluation results under the baseline/ directory in the shared task data.

To run the **./baseline.py** program stand-alone, you can use the following flags:

```
./baseline.py --task=TASK# --language=LANGUAGE [--align=mcmc|med|dumb] [--path=DATAPATH]
```

For example, running:

```
./baseline.py --task=1 --language=navajo --path=./../../data/
```

would solve task 1 for navajo, with the data files located in **./../../data/**, and assume the training file is called **navajo-task1-train** and that the evaluation file is called **navajo-task1-dev**.  The guesses are output to STDOUT in a format understood by [evalm.py](https://github.com/ryancotterell/sigmorphon2016/tree/master/src).

Note that the script, when solving task 1, uses only the training data for task 1.  However, when solving task 2, it also uses only the training data for task 1, but evaluates against task 2 dev data; when solving task 3, it uses the training data for all three tasks, but evaluates against task 3's dev data.

## Details

The system is a simple discriminative string transduction which uses an averaged perceptron as the classifier.  First, all word pairs in the training data are symbol-aligned 1-to-1 through an iterative Markov Chain Monte Carlo method.  After this, the alignments are chunked so that consecutive deletions, insertions, and changes become one single such action.  Then, a classifier is trained to make a decision when moving left-to-right through an input string.  Here is an example string pair from the Finnish data after alignment and chunking:

```
1 2 3 4 5 6 
k a t - o ssa   (source) pos=N,case=IN+ESS,num=SG
k a t t o -     (target) lemma
```

The correct decision, i.e. class, for positions 1,2,3,5 is **repeat**. For position 4 the decision is **insert t**, for position 6 the decision is **delete ssa**.  The classifier is trained using the following binary features at each position:

* The previous 1,2, and 3 input symbols; e.g. **t**,**at**,**kat** for pos 4 in the above.
* The previous 1,2, and 3 output symbols; e.g. **t**,**tt**,**att** for pos 5 in the above.
* The following 1,2, and 3 input symbols, e.g. **o**, **os**, **oss**, for pos 3 in the above.
* The previous action (**insert X**, **repeat**, **change X**, **delete N**), where **X** is a string and **N** is a number.  For example, **insert t** for position 5 in the above.
* The morphosyntactic features of the source (for lemmatization) or target form (for lemma → form mapping), e.g. **pos=N,case=IN+ESS,num=SG** for the above.
* Any combinations of two features from the above list where the first feature in the combination is a morphosyntactic feature and the second is not.

We construct one classifier per part-of-speech.  For task 1, a classifier for lemma to arbitrary word form is built.  For task 2, two classifiers are trained for each POS: one for lemmatization, and another for lemma → word form mapping.  These are then applied in series.

For task 3, an additional classifier is built to map a word to its morphosyntactic description (using training data from all tasks).  Task 3 is then solved by first classifying the input word form to yield a guess as to its morphosyntactic description, then lemmatizing it and mapping the lemma to the target form, as in task 2.

The baseline system performs no real tuning of parameters or feature selection.  The averaged perceptron is not trained with early stopping, and simply runs for 10 iterations or until the data are separated.
