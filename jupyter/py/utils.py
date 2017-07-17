def load_vocabs(fname):
    with open(fname, encoding='utf-8') as f:
        vocabs = [doc.strip() for doc in f]
    return vocabs

def load_x(fname):
    from scipy.io import mmread
    return mmread(fname).tocsr()

def asintarray(y, to_int=False):
    import numpy as np
    y = [1 if yi == 'N' else -1 for yi in y]
    y = np.asarray(y)
    return y

def load_data(head, directory):
    x = load_x('%s/%s_x.mtx' % (directory, head))
    y = asintarray(load_vocabs('%s/%s_y' % (directory, head)))
    x_word = load_vocabs('%s/%s_x_word' % (directory, head))
    vocabs = load_vocabs('%s/%s_vocabs' % (directory, head))

    print('x shape = %s\ny shape = %s\n# features = %d\n# L words = %d' % (x.shape, y.shape, len(vocabs), len(x_word)))
    return x, y, x_word, vocabs