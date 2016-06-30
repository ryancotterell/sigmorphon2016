#!/usr/bin/env python
"""
Baseline system for the SIGMORPHON 2016 Shared Task.

Solves tasks 1,2, and 3, evaluating on dev data and outputs guesses.

Author: Mans Hulden
Last Update: 11/29/2015
"""

from __future__ import print_function
import perceptron_c, align, codecs, sys, re, getopt

class MorphModel:
    def __init__(self):
        self.features   = {'tolemma':None, 'fromlemma':None}
        self.classes    = {'tolemma':None, 'fromlemma':None}
        self.classifier = {'tolemma':None, 'fromlemma':None}
        
class Morph:

    def __init__(self):
        self.models = {}
        self.msdfeatures = None
        self.msdclasses = None
        self.msdclassifier = None        

    def generate(self, word, featurestring, mode):
        """Generates an output string from an input word and target
            feature string. The 'mode' variable is either 'tolemma' or
            'fromlemma' """
        pos = re.match(r'pos=([^,]*)', featurestring).group(1)
        ins = ['<'] + list(word) + ['>']
        outs = []
        prevaction = 'None'
        position = 0
        while position < len(ins):            
            feats = list(train_get_surrounding_syms(ins, position, u'in_')) + \
               list(train_get_surrounding_syms(outs, position, u'out_', lookright = False)) + \
               ['prevaction='+prevaction] + [u'MSD:' + featurestring]
            feats = feature_pairs(feats)
            decision = self.models[pos].classifier[mode].decision_function(feats)
            decision = sorted(decision, key = lambda x: x[1], reverse = True)
            prevaction = self._findmax(decision, prevaction, len(ins)-position-1)
            actionlength, outstring = interpret_action(prevaction, ins[position])
            outs.append(outstring)
            position += actionlength
        return ''.join(outs[1:-1])
            
    def _findmax(self, decision, lastaction, maxlength):
        """Find best action that doesn't conflict with last (can't del/ins/chg two in a row)
           and isn't too long (can't change/del more than what remains)."""
        if lastaction[0] == 'D' or lastaction[0] == 'C' or lastaction[0] == 'I':
            for x in xrange(len(decision)):
                if decision[x][0][0] != lastaction[0]:
                    if decision[x][0][0] == u'C' and len(decision[x][0][1:]) > maxlength:
                        continue
                    if decision[x][0][0] == u'D' and int(decision[x][0][1:]) > maxlength:
                        continue
                    return decision[x][0]
        else:
            return decision[0][0]
            
    def add_features(self, pos, features, classes, mode):
        """Adds a collection of feautures and classes to a pos model
           'mode' is either 'tolemma' or 'fromlemma'."""
        if pos not in self.models:
            self.models[pos] = MorphModel()
        self.models[pos].features[mode] = features
        self.models[pos].classes[mode] = classes
        
    def get_pos(self):
        """Simply lists all poses associated with a model."""
        return list(self.models.keys())

    def add_classifier(self, pos, classifier, mode):
        """Adds a classifier to a pos model in a certain mode."""
        self.models[pos].classifier[mode] = classifier
        
    def get_features(self, pos, mode):
        return self.models[pos].features[mode]

    def get_classes(self, pos, mode):
        return self.models[pos].classes[mode]

    def extract_task3(self, lang, path):
        
        # We use the msd/form combinations from all three
        msdform = set()
        lines = [line.strip() for line in codecs.open(path + lang +'-task1-train', "r", encoding="utf-8")]
        for l in lines:
            lemma, msd, form = l.split(u'\t')
            msdform.add((msd, form))
        lines = [line.strip() for line in codecs.open(path + lang +'-task2-train', "r", encoding="utf-8")]
        for l in lines:
            msd1, form1, msd2, form2 = l.split(u'\t')
            msdform.add((msd1, form1))
            msdform.add((msd2, form2))
        lines = [line.strip() for line in codecs.open(path + lang +'-task3-train', "r", encoding="utf-8")]
        for l in lines:
            form1, msd2, form2 = l.split(u'\t')
            msdform.add((msd2, form2))

        self.msdfeatures = []
        self.msdclasses = []
        for msd, form in msdform:
            formfeatures = extract_substrings(form)
            self.msdfeatures.append(formfeatures)
            self.msdclasses.append(msd)
                
    def extract_task1(self, filename, mode, path):
        """Parse a file and extract features/classes for
        mapping to and from a lemma form."""
    
        lemmas = {}
        poses = set()
        lines = [line.strip() for line in codecs.open(path + filename, "r", encoding="utf-8")]
        for l in lines:
            if 'pos=' not in l:
                continue
            lemma, feats, form = l.split(u'\t')
            pos = re.match(r'pos=([^,]*)', feats).group(1)
            if lemma not in lemmas:
                lemmas[lemma] = []
                lemmas[lemma].append((lemma, 'pos=' + pos + ',lemma=true'))
            lemmas[lemma].append((form, feats))
            if pos not in poses:
                poses.add(pos)

        pairs = []
        wordpairs = []
        for lemma in lemmas:
            lemmafeatures = lemmas[lemma]
            for x in lemmafeatures:
                for y in lemmafeatures:
                    if (x != y) and ('lemma=true' in x[1]) and (mode == 'fromlemma'):
                        pairs.append(tuple((x[0], y[0], y[1])))
                        # inword, outword, msdfeatures
                        wordpairs.append(tuple((x[0], y[0])))
                    elif (x != y) and ('lemma=true' in x[1]) and (mode == 'tolemma'):
                        pairs.append(tuple((y[0], x[0], y[1])))
                        # inword, outword, msdfeatures
                        wordpairs.append(tuple((y[0], x[0])))

        if ALIGNTYPE == 'mcmc':
            alignedpairs = mcmc_align(wordpairs, ALIGN_SYM)
        elif ALIGNTYPE == 'med':
            alignedpairs = med_align(wordpairs, ALIGN_SYM)
        else:
            alignedpairs = dumb_align(wordpairs, ALIGN_SYM)
        
        chunkedpairs = chunk(alignedpairs)

        for pos in poses: # Do one model per POS
            features = []
            classes = []
            for idx, pair in enumerate(chunkedpairs):
                if 'pos=' + pos not in pairs[idx][2]:
                    continue
                instring = ['<'] + [x[0] for x in pair] + ['>']
                outstring = ['<'] + [x[1] for x in pair] + ['>']

                #msdfeatures = pairs[idx][2].split(':') # separate features 
                msdfeatures = [ pairs[idx][2] ] # don't separate features
                msdfeatures = ['MSD:' + f for f in msdfeatures]
                prevaction = 'None'
                for position in range(0, len(instring)):
                    thiscl, feats = train_get_features(instring, outstring, position)
                    classes.append(thiscl)
                    featurelist = list(feats) + msdfeatures + ['prevaction='+prevaction]
                    featurelist = feature_pairs(featurelist)
                    features.append(featurelist)
                    prevaction = thiscl
            self.add_features(pos, features, classes, mode)

def feature_pairs(f):
    """Expand features to include pairs of features 
    where one is always a f=v feature."""
    pairs = [x + ".x." + y for x in f for y in f if u'=' in y]
    return pairs + f
    
def dumb_align(wordpairs, align_symbol):
    alignedpairs = []
    for idx, pair in enumerate(wordpairs):
        ins = pair[0]
        outs = pair[1]
        if len(ins) > len(outs):
            outs = outs + align_symbol * (len(ins)-len(outs))
        elif len(outs) > len(ins):
            ins = ins + align_symbol * (len(outs)-len(ins))
            alignedpairs.append((ins, outs))
    return alignedpairs
    
def mcmc_align(wordpairs, align_symbol):
    a = align.Aligner(wordpairs, align_symbol = align_symbol, random_seed = 42)
    return a.alignedpairs
    
def med_align(wordpairs, align_symbol):
    a = align.Aligner(wordpairs, align_symbol = align_symbol, mode = 'med')
    return a.alignedpairs

def train_get_surrounding_syms(s, position, featureprefix, lookright = True):
    """Get surrounding symbols from a list of chunks and position.
    >>> s = ['<', u'a', u'b', u'u', u'_', u't', u'a', u'n', u'doka', '>']
    >>> train_get_surrounding_syms(s, 4, 'in_')
    set([u'nin_ta', u'nin_t', u'nin_tan', u'pin_u', u'pin_bu', u'pin_abu'])
    """
    leftfeats = set()
    rightfeats = set()
    if position == 0:
        leftfeats |= {u'p' + featureprefix + u'none'}
    if (position == len(s)) and lookright:
        rightfeats |= {u'n' + featureprefix + u'none'}
    if position > 0:
        left = ''.join(s[:position]).replace(u'_', u'')
        leftfeats |= {u'p' + featureprefix + left[x:] for x in [-1,-2,-3]}
    if (position < len(s)) and lookright:
        right = ''.join(s[position:]).replace(u'_', u'')
        rightfeats |= {u'n' + featureprefix + right[:x] for x in [1,2,3]}
    return leftfeats | rightfeats
    
def train_get_features(ins, outs, position):
    feats = set()
    # Get class first #
    if ins[position] == outs[position]:
        cl = "R"
    elif u'_' in ins[position]:
        cl = "I" + outs[position]
    elif u'_' in outs[position]:
        cl = "D" + unicode(len(ins[position]))
    else:
        cl = "C" + outs[position]
        
    # Get features of surrounding symbols #
    feats |= train_get_surrounding_syms(ins, position, u'in_')
    feats |= train_get_surrounding_syms(outs, position, u'out_', lookright = False)
    return cl, feats

def interpret_action(action, ins):
    """Interpret classifier class: return length of input to consume + output."""
    if action[0] == u'R':
        return (1, ins)
    elif action[0] == u'D':
        return int(action[1:]), u''
    elif action[0] == u'C':
        return len(action[1:]), action[1:]
    elif action[0] == u'I':
        return 0, action[1:]
    
def chopup(s, t):
    """Returns grouped alignment of two strings
       in such a way that consecutive del/ins/chg operations
       are grouped to be one single operation.
       The input is two 1-to-1 aligned strings where _ = empty string.
    >>> chopup(['ka__yyab','kaxx__xy'])
    (['k', 'a', u'_', 'yy', 'ab'], ['k', 'a', 'xx', u'_', 'xy'])
    """
    def action(inchar, outchar):
        if inchar == u'_':
            return 'ins'
        elif outchar == u'_':
            return 'del'
        elif inchar != outchar:
            return 'chg'
        else:
            return 'rep'
            
    idx = 1
    s = list(s)
    t = list(t)
    while idx < len(s):
        l = action(s[idx-1], t[idx-1])
        r = action(s[idx], t[idx])
        if (l == 'rep' and r == 'rep') or (l != r):
            s.insert(idx, ' ')
            t.insert(idx, ' ')
            idx += 1
        idx += 1
    s = tuple(u'_' if u'_' in x else x for x in ''.join(s).split(' '))
    t = tuple(u'_' if u'_' in x else x for x in ''.join(t).split(' '))
    return zip(s,t)
    
def chunk(pairs):
    """Chunk alignments to have possibly more than one symbol-one symbol."""
    chunkedpairs = []
    for instr, outstr in pairs:
        chunkedpairs.append(chopup(instr, outstr))
    return chunkedpairs
          
def extract_substrings(word):
    """Get len 2/3 substrings and return as list."""
    w3 = zip(word, word[1:], word[2:])
    w2 = zip(word, word[1:])
    return [''.join(x) for x in w2+w3]

def announce(*objs):
    print("***", *objs, file = sys.stderr)
    
def main(argv):
    global ALIGN_SYM
    global ALIGNTYPE
    global TASK
    
    options, remainder = getopt.gnu_getopt(argv[1:], 'l:t:a:p:', ['language=','task=','align=','path='])

    PATH, ALIGN_SYM, ALIGNTYPE, TASK = './', u'_', 'mcmc', 1
    for opt, arg in options:
        if opt in ('-l', '--language'):
            LANGUAGE = arg
        elif opt in ('-t', '--task'):
            TASK = int(arg)
        elif opt in ('-a', '--align'):
            ALIGNTYPE = arg
        elif opt in ('-p', '--path'):
            PATH = arg

    train = Morph()
    announce(LANGUAGE + ": learning alignment for form > lemma mapping")
    train.extract_task1(LANGUAGE + '-task1-train', 'fromlemma', PATH)
    if TASK == 2 or TASK == 3:
        announce(LANGUAGE + ": learning alignment for lemma > form mapping")
        train.extract_task1(LANGUAGE + '-task1-train', 'tolemma', PATH)

    if TASK == 1 or TASK == 2 or TASK == 3:
        for pos in train.get_pos():
            announce(LANGUAGE + ": training " + pos + " for lemma > form mapping")
            P = perceptron_c.Perceptron(shuffle = True, averaged = True, verbose = True, max_iter = 10, random_seed = 42)
            P.fit(train.get_features(pos, 'fromlemma'), train.get_classes(pos, 'fromlemma'))
            train.add_classifier(pos, P, 'fromlemma')

    if TASK == 2 or TASK == 3:
        for pos in train.get_pos():
            announce(LANGUAGE + ": training " + pos + " for form > lemma mapping")
            P = perceptron_c.Perceptron(shuffle = True, averaged = True, verbose = True, max_iter = 10, random_seed = 42)
            P.fit(train.get_features(pos, 'tolemma'), train.get_classes(pos, 'tolemma'))
            train.add_classifier(pos, P, 'tolemma')

    if TASK == 3:
        train.extract_task3(LANGUAGE, PATH)
        announce(LANGUAGE + ": training form > msd classifier")
        train.msdclassifier = perceptron_c.Perceptron(shuffle = True, averaged = True, verbose = True, max_iter = 10, random_seed = 42)
        train.msdclassifier.fit(train.msdfeatures, train.msdclasses)
        
    testlines = [line.strip() for line in codecs.open(PATH+LANGUAGE + '-task' + str(TASK) + '-dev', "r", encoding="utf-8")]
    if TASK == 1:
        for l in testlines:
            lemma, targetmsd, wordform = l.split('\t')
            guess = train.generate(lemma, targetmsd, 'fromlemma')
            print((lemma + "\t" + targetmsd + "\t" + guess).encode("utf-8"))
            
    if TASK == 2:
        for l in testlines:
            sourcemsd, sourceform, targetmsd, targetform = l.split('\t')
            lemma = train.generate(sourceform, sourcemsd, 'tolemma')
            guess = train.generate(lemma, targetmsd, 'fromlemma')
            print((sourcemsd + "\t" + sourceform + "\t" + targetmsd + "\t" + guess).encode("utf-8"))

    if TASK == 3:
        for l in testlines:
            sourceform, targetmsd, targetform = l.split('\t')
            sourcemsd = train.msdclassifier.predict(extract_substrings(sourceform))
            lemma = train.generate(sourceform, sourcemsd, 'tolemma')
            guess = train.generate(lemma, targetmsd, 'fromlemma')
            print((sourceform + "\t" + targetmsd + "\t" + guess).encode("utf-8"))
            
if __name__ == "__main__":
    main(sys.argv)
