import sys
from soynlp.utils import get_process_memory
from soynlp.hangle import normalize
import numpy as np

class TrainedNounExtractor:
    def __init__(self, coefficient, max_length=8):
        self._coef = coefficient
        self.lmax = max_length
        self.josa_threshold = -0.1
        self.josa = sorted(filter(lambda x:x[1] > self.josa_threshold, self._coef.items()), key=lambda x:x[1], reverse=True)
        self.postnoun = set('가감계과권꾼네님대댁당들력로론류률상생성시식용율의인이일재적제중째쯤층치파판풍형화')
        
    def extract(self, sents, min_count=10, min_noun_score=0.1):
        self.lrgraph, self.lset, self.rset = self._build_lrgraph(sents, min_count)
        self.lentropy, self.rentropy, self.lav, self.rav = self._branching_entropy(self.lrgraph)
        scores = self._compute_noun_score(self.lrgraph)
        scores = self._postprocessing(scores, min_noun_score)
        return scores

    def _build_lrgraph(self, sents, min_count, pruning_min_count=2):
        lset = {}
        rset = {}
        for n_sent, sent in enumerate(sents):
            for eojeol in sent.split():
                for e in range(1, min(len(eojeol), self.lmax)+1):
                    l = eojeol[:e]
                    r = eojeol[e:]
                    lset[l] = lset.get(l, 0) + 1
                    rset[r] = rset.get(r, 0) + 1
            if n_sent % 1000 == 999:
                args = (n_sent+1, len(lset), len(rset), get_process_memory())
                sys.stdout.write('\rscaning vocabulary ... %d sents #(l= %d, r= %d), mem= %.3f Gb' % args)
            if n_sent % 500000 == 499999:
                lset = {l:f for l,f in lset.items() if f >= pruning_min_count}
                rset = {l:f for l,f in rset.items() if f >= pruning_min_count}
        lset = {l:f for l,f in lset.items() if f >= min_count}
        rset = {l:f for l,f in rset.items() if f >= min_count}
        
        n_sents = n_sent
        
        lrgraph = {}
        for n_sent, sent in enumerate(sents):
            for eojeol in sent.split():
                for e in range(1, min(len(eojeol), self.lmax)+1):
                    l = eojeol[:e]
                    r = eojeol[e:]
                    if not (l in lset) or not (r in rset):
                        continue
                    rdict = lrgraph.get(l, {})
                    rdict[r] = rdict.get(r, 0) + 1
                    lrgraph[l] = rdict            
            if n_sent % 1000 == 999:
                args = (100*(n_sent+1)/n_sents, '%', n_sent+1, n_sents, get_process_memory())
                sys.stdout.write('\rbuilding lrgraph ... (%.3f %s, %d in %d), mem= %.3f Gb' % args)
        args = (len(lset), len(rset), sum((len(rdict) for rdict in lrgraph.values())), get_process_memory())
        print('\rlrgraph has been built. (#L= %d, #R= %d, #E=%d), mem= %.3f Gb' % args)
        return lrgraph, lset, rset
    
    def _branching_entropy(self, lrgraph):
        from collections import defaultdict
        def entropy_and_accessorvariety(d):
            sum_ = sum(d.values())
            if sum_ == 0: return (0, 0)
            return (-1 * sum((v/sum_) * np.log(v/sum_) for v in d.values()), len(d))
        def branch_map(d, get_branch):
            b = defaultdict(lambda: 0)
            for ext,f in d.items():
                if ext != '':
                    b[get_branch(ext)] += f
            return b
        def all_entropy(graph, get_branch=lambda x:x[0]):
            return {w:entropy_and_accessorvariety(branch_map(d, get_branch)) for w, d in graph.items()}
        def to_rlgraph(lrgraph):
            rlgraph = defaultdict(lambda: defaultdict(lambda: 0))
            for l, rdict in lrgraph.items():
                for r, f in rdict.items():
                    if r == '': continue
                    rlgraph[r][l] += f
            return {r:dict(ldict) for r, ldict in rlgraph.items()}
        print('compute branching entropy ...', end='')
        lvalues = all_entropy(lrgraph)
        rvalues = all_entropy(to_rlgraph(lrgraph), get_branch=lambda x:x[-1])
        lentropy = {w:v[0] for w,v in lvalues.items()}
        rentropy = {w:v[0] for w,v in rvalues.items()}
        lav = {w:v[1] for w,v in lvalues.items()}
        rav = {w:v[1] for w,v in lvalues.items()}
        print(' done')
        return lentropy, rentropy, lav, rav
        
    def _compute_noun_score(self, lrgraph):
        from collections import namedtuple
        Score = namedtuple('Score', 'score frequency be av p_feature p_eojeol')
        scores = {}
        n = len(lrgraph)
        for i, (l, rdict) in enumerate(lrgraph.items()):
            rdict_ = {r:f for r,f in rdict.items() if r in self._coef}
            rsum = sum((f for r,f in rdict.items() if r != ''))
            frequency = self.lset.get(l, 0)
            feature_fraction = sum(rdict_.values()) / rsum if rsum > 0 else 0
            eojeol_fraction = rdict.get('', 0) / frequency
            if not rdict_:
                score = 0
            else:
                score = sum(f*self._coef[r] for r, f in rdict_.items()) / sum(rdict_.values())
            scores[l] = Score(score, frequency, self.lentropy.get(l, 0), self.lav.get(l,0), feature_fraction, eojeol_fraction)
            if (i+1) % 1000 == 0:
                args = (100*(i+1)/n, '%', i+1, n)
                sys.stdout.write('\rcompute noun score ... (%.3f %s, %d in %d)' % args)
        print('\rcomputing noun score has been done.')
        return scores
#         return sorted(scores.items(), key=lambda x:x[1].score, reverse=True)
        
    def _postprocessing(self, scores, min_noun_score):
        nouns = self._nsubjsub_processing(scores, min_noun_score)
        nouns = self._compound_processing(nouns)
        return nouns
        
    def _nsubjsub_processing(self, scores, min_noun_score):
        def is_NsubJ(s_, s, bes_=0.5, f_fraction=0.5, eojeol_fraction_s_=0.1):
            return (s_.be < bes_) and (s.frequency / s_.frequency > f_fraction) and (s_.p_eojeol < eojeol_fraction_s_)
        def is_NJsub(s_, s, bes_=0.5, bes=0.5, eojeol_fraction_s=0.5, f_fraction=0.3):
            return (s_.be > bes_) and (s.be < bes or s.p_eojeol > eojeol_fraction_s)

        candidates = dict(filter(lambda x:x[1].score > min_noun_score, scores.items()))
        nouns = {}
        for w, s in sorted(candidates.items(), key=lambda x:len(x[0])):
            n = len(w)
            if n == 1:
                nouns[w] = s
                continue
            # Case: 떡볶 + 이 --> remove 떡볶
            if (w[-1] in self.postnoun) and (w[:-1] in nouns):
                s_ = nouns[w[:-1]]
                if is_NsubJ(s_, s):
                    del nouns[w[:-1]]

            # Case: [대학생 + 과] + 의 --> pass 대학생과 e = 1
            # Case: [정국 + 으로] + 의 --> pass 정국으로 e = 2
            NJsub = False
            for e in range(1, max(5, n)):
                l = w[:-e]
                jsub = w[-e:]
                if (l in nouns) and (jsub in self.josa):
                    s_ = nouns[l]
                    if is_NJsub(s_, s):
                        NJsub = True
                        break
                    if 0 == sum((1 if (jsub+r) in self.josa else 0 for r in self.lrgraph.get(l, {}).keys())):
                        NJsub = True
                        break
            if NJsub:
                continue
            nouns[w] = s
        print('n_candidates= %d' % len(candidates))
        print('n_nouns after substring processing= %d' % len(nouns))
        return nouns
        
    def _compound_processing(self, nouns):
        def match_r_is_nounjosa(base, next_character, nouns):
            def match(l, r):
                if (not r) and ((l in nouns) and not (l in self.josa)): return True
                return ((l in nouns) and not (l in self.josa)) and (r in self.josa)
            matched = []
            if not (base in nouns): return matched
            for r, f in  self.lrgraph.get(base, {}).items():
                if not r or r[0] != next_character: continue
                if r in self.postnoun: matched.append((r, ''))
                n = len(r)
                if n < 2:continue
                for i in range(2, n+1):
                    ls = r[:i]
                    rs = r[i:]
                    if match(ls, rs):
                        matched.append((ls, rs))
            return matched
        def get_branch(base):
            return {r for r in self.lrgraph.get(base, {}).keys() if r}
        droprate = lambda shorter,longer: 1 if not shorter in self.lset else 1 - (self.lset.get(longer, 0) / self.lset[shorter])
        
        droprate_threshold = 0.8
        postprocess_noun_threshold=0.5
        postprocess_be_threshold = 0.8

        compound_processeds = {}
        for noun, score in nouns.items():
            if (len(noun) <= 2) or (droprate(noun[:-1], noun) < droprate_threshold) or (score.be > postprocess_be_threshold): 
                compound_processeds[noun] = score
                continue
            noun_sub = noun[:-1]
            matched = match_r_is_nounjosa(noun_sub, noun[-1], nouns)
            if matched:
                for l1 in set(tuple(zip(*matched))[0]):
                    compound = noun_sub+l1
                    if (compound in self.lrgraph) and (len(get_branch(compound)) > 2):
                        compound_processeds[compound] = (noun_sub, l1)
                continue
            if (score.score > postprocess_noun_threshold) and (score.be > postprocess_be_threshold):
                compound_processeds[noun] = score
        print('n_nouns after compound processing= %d' % len(compound_processeds))
        return compound_processeds