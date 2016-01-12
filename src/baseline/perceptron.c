/************************************************************************/
/* Simple perceptron/averaged perceptron library                        */
/* Author: Mans Hulden (mans.hulden@gmail.com)                          */
/* Copyright 2014 Mans Hulden                                           */
/*                                                                      */
/* Licensed under the Apache License, Version 2.0 (the "License");      */
/* you may not use this file except in compliance with the License.     */
/* You may obtain a copy of the License at                              */
/*                                                                      */
/*     http://www.apache.org/licenses/LICENSE-2.0                       */
/*                                                                      */
/* - MH20140831                                                         */
/************************************************************************/

/* To build for python bindings: gcc -O3 -Wall -Wextra -shared perceptron.c -o libperceptron.so */

/* Usage:

(1) Call perceptron_init with desired parameters
(2) Add examples to training set using examples_add()
    - optionally also to dev set using devexamples_add()
(3) Train perceptron using perceptron_train()
(4) Classify with perceptron_decision_function_int()/perceptron_decision_function_double()
    (int is for non-averaged/double for averaged)
    which return a vector of weights for all classes
    or:
    Just use perceptron_classify_double()/perceptron_classify_int()
    which return the best class index
(5) perceptron_destroy() frees all data structures associated

Notes:

- Only supports binary features in examples
- Uses a sparse representation where only hot features are given for examples
- Weights are integers, although double weights are used for the averaged case

*/

/******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>  /* For INT_MAX */
#include <float.h>   /* For DBL_MAX */

struct perceptron {
    struct examples *ex;     /* Training examples */
    struct examples *devex;  /* Dev examples      */
    int averaged;            /* Use averaged perceptron or vanilla? */
    int tune_on_averaged;    /* Whether to tune AP on averaged weights or running weights */
    int num_examples;        /* Training set size */
    int num_devexamples;     /* Dev set size      */
    int examplecounter;      /* Running counter when adding examples one-by-one */
    int devexamplecounter;   /* Running counter when adding examples one-by-one */
    int num_classes;         /* Number of distinct classes */
    int num_features;        /* Numer of features */
    int max_iter;            /* Max iterations to run */
    int shuffle;             /* Shuffle examples before each iteration? */
    int verbose;             /* Print stats to stderr? */

    int *intweights;         /* Running int weights */
    int *intbiases;          /* Running int biases  */
    double *doubleweights;   /* Weights for averaged perceptron */
    double *doublebiases;    /* Biases */
    double *lastdweights;    /* Store temp weights for tuning w/ dev set */
    double *lastdbiases;     /* Store temp biases for tuning w/ dev set */
    int *lastiweights;       /* Store temp weights for tuning w/ dev set */
    int *lastibiases;        /* Store temp biases for tuning w/ dev set */
};

struct examples {
    int *hotfeatures; /* A list of the features that are hot in this example */
    int len;          /* Number of hot features in example */
    int correctclass; /* The class the example belongs to */
};

/******************************************************************************/

/* Initialize the perceptron structure, returns handle */
struct perceptron *perceptron_init(int max_iter, int num_examples, int num_devexamples, int num_features, int num_classes, int averaged, int shuffle, int random_seed, int tune_on_averaged, int verbose);

/* Train perceptron w/ current training/(dev) examples and settings */
void perceptron_train(struct perceptron *perceptron);

/* Free examples/weights + perceptron data structure */
void perceptron_destroy(struct perceptron *p);

/* Decision function for example (features holds list of hot features, len is number of hot features) */
/* Returns vector of weights for each class (highest weight is best class) */
double *perceptron_decision_function_double(struct perceptron *perceptron, int *features, int len);

/* Decision function for non-averaged perceptron */
/* Returns vector of weights for each class (highest weight is best class) */
int *perceptron_decision_function_int(struct perceptron *perceptron, int *features, int len);

/* Classify function for example (features holds list of hot features, len is number of hot features) */
/* Returns best class number */
/* dev_tuning and numiter are internal parameters used while training/set these to 0,0 */
int perceptron_classify_double(struct perceptron *perceptron, int *features, int len, int dev_tuning, int numiter);

/* Classify function for example (features holds list of hot features, len is number of hot features) */
/* Returns best class number */
/* Use for non-averaged perceptron */
int perceptron_classify_int(struct perceptron *perceptron, int *features, int len);

/* Add an example to the training set */
/* supply perceptron handle, a vector of hot features, len of this vector, and correct class index */
void examples_add(struct perceptron *perceptron, int *features, int len, int correctclass);

/* Add an example to the dev set */
/* supply perceptron handle, a vector of hot features, len of this vector, and correct class index */
void devexamples_add(struct perceptron *perceptron, int *features, int len, int correctclass);

/******************************************************************************/

struct perceptron *perceptron_init(int max_iter, int num_examples, int num_devexamples, int num_features, int num_classes, int averaged, int shuffle, int random_seed, int tune_on_averaged, int verbose) {
    struct perceptron *p;
    p = calloc(1, sizeof(struct perceptron));
    p->max_iter = max_iter;
    p->num_examples = num_examples;
    p->examplecounter = 0;
    p->devexamplecounter = 0;
    p->num_classes = num_classes;
    p->num_features = num_features;
    p->shuffle = shuffle;
    p->ex = calloc(num_examples, sizeof(struct examples));
    p->verbose = verbose;
    p->tune_on_averaged = tune_on_averaged;
    if (random_seed)
	srand(random_seed);
    if (num_devexamples > 0) {
	p->num_devexamples = num_devexamples;
	p->devex = calloc(num_devexamples, sizeof(struct examples));
    }
    p->intweights = calloc(num_features * num_classes, sizeof(int));
    p->intbiases = calloc(num_classes, sizeof(int));
    p->averaged = averaged;
    if (p->averaged) {
	p->doubleweights = calloc(num_features * num_classes, sizeof(double));
	p->doublebiases = calloc(num_classes, sizeof(double));
	p->lastdweights = calloc(num_features * num_classes, sizeof(double));
	p->lastdbiases = calloc(num_classes, sizeof(double));
	p->lastiweights = calloc(num_features * num_classes, sizeof(int));
	p->lastibiases = calloc(num_classes, sizeof(int));
    }
    return p;
}

static int rand_int(int n) {
    int limit = RAND_MAX - RAND_MAX % n;
    int rnd;
    do {
	rnd = rand();
    } while (rnd >= limit);
    return rnd % n;
}

void shuffle(int *array, int n) {
    int i, j, tmp;
    for (i = n - 1; i > 0; i--) {
	j = rand_int(i + 1);
	tmp = array[j];
	array[j] = array[i];
	array[i] = tmp;
    }
}

void perceptron_train(struct perceptron *perceptron) {
    int i, j, n, m, guessedclass, correctclass, *weightptr, numincorrect, itercount, *classorder, devcorrect, devlastcorrect;
    double *dweightptr;
    itercount = 1;
    classorder = calloc(perceptron->num_examples, sizeof(int));
    for (i = 0; i < perceptron->num_examples; i++) {
	classorder[i] = i;
    }
    devlastcorrect = 0;
    for (i = 0; i < perceptron->max_iter; i++) {
	if (perceptron->shuffle)
	    shuffle(classorder, perceptron->num_examples);
	numincorrect = 0;
	for (n = 0; n < perceptron->num_examples; n++) {
	    m = classorder[n];
	    guessedclass = perceptron_classify_int(perceptron, perceptron->ex[m].hotfeatures, perceptron->ex[m].len);
	    correctclass = perceptron->ex[m].correctclass;
	    if (guessedclass != correctclass) {
		numincorrect++;
		for (j = 0; j < perceptron->ex[m].len; j++) {
		    weightptr = perceptron->intweights + perceptron->num_features * correctclass; /* Points to correct class weights */
		    weightptr += *(perceptron->ex[m].hotfeatures + j);
		    *(weightptr) += 1;
		    
		    weightptr = perceptron->intweights + perceptron->num_features * guessedclass; /* Points to incorrect class weights */
		    weightptr += *(perceptron->ex[m].hotfeatures + j);
		    *(weightptr) -= 1;		    
		}
		perceptron->intbiases[correctclass] += 1;
		perceptron->intbiases[guessedclass] -= 1;
 		if (perceptron->averaged) {
		    for (j = 0; j < perceptron->ex[m].len; j++) {
			dweightptr = perceptron->doubleweights + perceptron->num_features * correctclass; /* Points to correct class weights */
			dweightptr += *(perceptron->ex[m].hotfeatures + j);
			*(dweightptr) += 1.0 * (double)itercount;

			dweightptr = perceptron->doubleweights + perceptron->num_features * guessedclass; /* Points to incorrect class weights */
			dweightptr += *(perceptron->ex[m].hotfeatures + j);
			*(dweightptr) -= 1.0 * (double)itercount;
		    }
		    perceptron->doublebiases[correctclass] += 1.0 * (double)itercount;
		    perceptron->doublebiases[guessedclass] -= 1.0 * (double)itercount;
		}
	    }
	    itercount++;	   
	}

	/* Print stats */
	if (perceptron->verbose) {
	    fprintf(stderr, "Iteration %i - TRAIN: (%i/%i) %lg", i+1, perceptron->num_examples-numincorrect, perceptron->num_examples, (double)(perceptron->num_examples-numincorrect)/(double)perceptron->num_examples);
	}
	/* Now test on dev set (if available) */
	if (perceptron->num_devexamples > 0) {
	    for (n = 0, devcorrect = 0; n < perceptron->num_devexamples; n++) {
		if (perceptron->averaged && perceptron->tune_on_averaged) {
		    guessedclass = perceptron_classify_double(perceptron, perceptron->devex[n].hotfeatures, perceptron->devex[n].len, 1, itercount);
		} else {
		    guessedclass = perceptron_classify_int(perceptron, perceptron->devex[n].hotfeatures, perceptron->devex[n].len);
		}
		correctclass = perceptron->devex[n].correctclass;
		if (guessedclass == correctclass) {
		    devcorrect++;
		}
	    }
	    if (perceptron->verbose)
		fprintf(stderr, " - DEV (%i/%i) %lg", devcorrect, perceptron->num_devexamples, (double)devcorrect/(double)perceptron->num_devexamples);
	    if (devcorrect < devlastcorrect) {
		if (perceptron->verbose)
		    fprintf(stderr, "\n");
		break; /* Stop iterations - performance went down */
	    }
	    devlastcorrect = devcorrect;
	} 
	if (perceptron->verbose)
	    fprintf(stderr, "\n");
	if (numincorrect == 0) {
	    break;
	}
	/* Store current (averaged) weights so we can restore them if performance goes down */
	if (perceptron->averaged) {
	    memcpy(perceptron->lastdweights, perceptron->doubleweights, perceptron->num_features * perceptron->num_classes * sizeof(double));
	    memcpy(perceptron->lastdbiases, perceptron->doublebiases, perceptron->num_classes * sizeof(double));
	    memcpy(perceptron->lastiweights, perceptron->intweights, perceptron->num_features * perceptron->num_classes * sizeof(int));
	    memcpy(perceptron->lastibiases, perceptron->intbiases, perceptron->num_classes * sizeof(int));
	}
    }
    if (perceptron->averaged) {
	/* If we use AP w/ dev set, take previous weights because performance has dropped on dev set */
	for (i = 0; i < perceptron->num_features * perceptron->num_classes; i++) {
	    if (perceptron->num_devexamples > 0 && perceptron->tune_on_averaged)
		perceptron->doubleweights[i] = (double)perceptron->lastiweights[i] - perceptron->lastdweights[i]/((double)itercount - 1);
	    else
		perceptron->doubleweights[i] = (double)perceptron->intweights[i] - perceptron->doubleweights[i]/(double)itercount;
	}
	for (i = 0; i < perceptron->num_classes; i++) {
	    if (perceptron->num_devexamples > 0 && perceptron->tune_on_averaged)
		perceptron->doublebiases[i] = (double)perceptron->lastibiases[i] - perceptron->lastdbiases[i]/((double)itercount - 1);
	    else 
		perceptron->doublebiases[i] = (double)perceptron->intbiases[i] - perceptron->doublebiases[i]/(double)itercount;
	}
    }
}

void perceptron_free_wrapper(void *ptr) {
    if (ptr != NULL)
	free(ptr);
}

void perceptron_destroy(struct perceptron *p) {
    free(p->ex);
    if (p->num_devexamples > 0) {
	free(p->devex);
    }
    free(p->intweights);
    free(p->intbiases);
    if (p->averaged) {
	free(p->doubleweights);
	free(p->doublebiases);
	free(p->lastdweights);
	free(p->lastdbiases);
	free(p->lastiweights);
	free(p->lastibiases);
    }
    free(p);
}

double *perceptron_decision_function_double(struct perceptron *perceptron, int *features, int len) {
    int f, c, fnum;
    double cumweight, *fweight, *prediction;
    prediction = calloc(perceptron->num_classes, sizeof(double));
    for (c = 0; c < perceptron->num_classes; c++) {
	cumweight = 0.0;
	for (f = 0; f < len; f++) {
	    fnum = features[f]; /* Feature that is hot */
	    fweight = perceptron->doubleweights + perceptron->num_features * c + fnum;
	    cumweight += *fweight;
	}
	cumweight += perceptron->doublebiases[c];
	prediction[c] = cumweight;
    }
    return prediction;
}

int *perceptron_decision_function_int(struct perceptron *perceptron, int *features, int len) {
    int f, c, fnum;
    int cumweight, *fweight, *prediction;
    prediction = calloc(perceptron->num_classes, sizeof(int));
    for (c = 0; c < perceptron->num_classes; c++) {
	cumweight = 0;
	for (f = 0; f < len; f++) {
	    fnum = features[f]; /* Feature that is hot */
	    fweight = perceptron->intweights + perceptron->num_features * c + fnum;
	    cumweight += *fweight;
	}
	cumweight += perceptron->intbiases[c];
	prediction[c] = cumweight;
    }
    return prediction;
}

int perceptron_classify_double(struct perceptron *perceptron, int *features, int len, int dev_tuning, int numiter) {
    int f, c, fnum, maxclass, ptr;
    double maxweight, cumweight, fweight;
    maxclass = 0;
    maxweight = -DBL_MAX;
    for (c = 0; c < perceptron->num_classes; c++) {
	cumweight = 0.0;
	for (f = 0; f < len; f++) {
	    fnum = features[f]; /* Feature that is hot */
	    if (dev_tuning) {
		ptr = perceptron->num_features * c + fnum;
		fweight = (double)perceptron->intweights[ptr] - perceptron->doubleweights[ptr]/(double)numiter;
	    } else {
		fweight = perceptron->doubleweights[perceptron->num_features * c + fnum];
	    }
	    cumweight += fweight;
	}
	if (dev_tuning) {
	    cumweight += (double)perceptron->intbiases[c] - perceptron->doublebiases[c]/(double)numiter;
	} else {
	    cumweight += perceptron->doublebiases[c];
	}
	if (cumweight > maxweight) {
	    maxweight = cumweight;
	    maxclass = c;
	}
    }
    return maxclass;
}

int perceptron_classify_int(struct perceptron *perceptron, int *features, int len) {
    int f, c, *fweight, fnum, maxclass, maxweight, cumweight;    
    maxclass = 0;
    maxweight = -INT_MAX;
    for (c = 0; c < perceptron->num_classes; c++) {
	cumweight = 0;
	for (f = 0; f < len; f++) {
	    fnum = features[f]; /* Feature that is hot */
	    fweight = perceptron->intweights + perceptron->num_features * c + fnum;
	    cumweight += *fweight;
	}
	cumweight += perceptron->intbiases[c];
	if (cumweight > maxweight) {
	    maxweight = cumweight;
	    maxclass = c;
	}
    }
    return maxclass;
}

void examples_add(struct perceptron *perceptron, int *features, int len, int correctclass) {
    struct examples *ex;
    ex = perceptron->ex;
    ex[perceptron->examplecounter].len = len;
    ex[perceptron->examplecounter].correctclass = correctclass;
    ex[perceptron->examplecounter].hotfeatures = malloc(len * sizeof(int));
    memcpy(ex[perceptron->examplecounter].hotfeatures, features, len * sizeof(int));
    perceptron->examplecounter++;
}

void devexamples_add(struct perceptron *perceptron, int *features, int len, int correctclass) {
    struct examples *ex;
    ex = perceptron->devex;
    ex[perceptron->devexamplecounter].len = len;
    ex[perceptron->devexamplecounter].correctclass = correctclass;
    ex[perceptron->devexamplecounter].hotfeatures = malloc(len * sizeof(int));
    memcpy(ex[perceptron->devexamplecounter].hotfeatures, features, len * sizeof(int));
    perceptron->devexamplecounter++;
}
