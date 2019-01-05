import numpy as np
import matplotlib.pyplot as plot


def main():
    # path to file 
    INPUT_FILE_PATH = '/homework/stochastic-course/task1/input.txt'
    #INPUT_FILE_PATH = 'input.txt'

    # output file path
    RESULT_PATH = '/homework/stochastic-course/task1/result.pdf'
    #RESULT_PATH = 'result.pdf'


    data = np.loadtxt(INPUT_FILE_PATH, delimiter='\t\t', skiprows=1)

    x = data[:, 0]
    y = data[:, 1]
    
    b_coef = np.ones((x.size,), dtype=int)
    a = np.column_stack((b_coef, x))
    aT = np.matrix.transpose(a)
    aT_a = aT.dot(a)
    (b, k) = np.linalg.pinv(aT_a).dot(aT).dot(y)

    x_min = min(x)
    x_max = max(x) 
    plot.plot([x_min, x_max], [k * x_min + b, k * x_max + b])
    plot.scatter(x, y, color='g')
    plot.savefig(RESULT_PATH)

    print('DONE!')

if __name__ == '__main__':
    main()