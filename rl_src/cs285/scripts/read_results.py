import glob
import tensorflow as tf
import os

def get_section_results(file):
    """
        requires tensorflow==1.12.0
    """
    X = []
    Y = []
    # for e in tf.train.summary_iterator(file):
    for e in tf.compat.v1.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag == 'Train_AverageReturn':
                X.append(v.simple_value)
            # elif v.tag == 'Eval_AverageReturn':
            elif v.tag == "Eval_AverageReturn":
                Y.append(v.simple_value)
    return X, Y

if __name__ == '__main__':
    import glob

    search = 'data/hw4_q5*random*/events*'
    logdir = os.path.normpath(os.path.join(os.getcwd(), search))
    eventfile = glob.glob(logdir)[0]

    X, Y = get_section_results(eventfile)
    for i, (x, y) in enumerate(zip(X, Y)):
        print('Iteration {:d} | Train_AverageReturn: {} | Eval_AverageReturn: {}'.format(i, x, y))