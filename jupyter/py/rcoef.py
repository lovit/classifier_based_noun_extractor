def lr_to_coefficient(X, y, vocabs):
    from collections import namedtuple
    Info = namedtuple('Info', 'coef count')
    X_frequency = [v[0] for v in X.sum(axis=1).tolist()]
    n_pos = sum([v for v, label in zip(X_frequency, y) if label == 1])
    n_neg = sum([v for v, label in zip(X_frequency, y) if label == -1])
    print('(pos= %.3f, neg= %.3f)' % (n_pos/(n_pos+n_neg), n_neg/(n_pos+n_neg)))
    
    X_pos = X[[i for i, label in enumerate(y) if label == 1]]
    X_neg = X[[i for i, label in enumerate(y) if label == -1]]
    feature_pos_frequency = X_pos.sum(axis=0).tolist()[0]
    feature_neg_frequency = X_neg.sum(axis=0).tolist()[0]
    
    vocab_frequency = X.sum(axis=0).tolist()[0]
    neg_factor = n_pos / n_neg
    score = lambda p,n: (p - n * neg_factor) / (p + n * neg_factor)
    coefficient = [score(p,n) for p, n in zip(feature_pos_frequency, feature_neg_frequency)]
    coefficient = {vocab:Info(coef, count) for count, vocab, coef in zip(vocab_frequency, vocabs, coefficient)}
    return coefficient