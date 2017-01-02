import numpy as np
import matplotlib.pyplot as plt


def g():
    num_input = 500
    offset = 3.0
    angle = np.pi / 3
    v1_v2 = np.random.randn(num_input, 2) * np.array([1, 0.5])
    rotate = np.array([[np.cos(angle), np.sin(angle)], [np.sin(angle), np.cos(angle)]])

    move = 2.0 - 2.0
    t1_t2 = np.dot(v1_v2, rotate) + np.array([move, move * np.sqrt(3.0)])

    print np.shape(t1_t2)
    [X, Y] = t1_t2.T
    plt.scatter(X, Y, alpha=.5)
    plt.show()

    
    with open("2D_gaussian_dist.txt", 'w') as fout:
    #with open("2D_gaussian_dist_shrink-test.txt", 'w') as fout:
    #with open("2D_gaussian_dist_small_var_positive.txt", 'w') as fout:
        for point in t1_t2:
            fout.write(str(offset) + ' ' + str(offset+point[0]) + ' ' + str(offset+point[1]) + '\n')

def l():
    num_input = 500
    offset = 3.0
    angle = np.pi / 2
    r = np.random.uniform(-0.8, 0.8, (num_input))
    v1_v2 = [[x, 0] for x in r]
    rotate = np.array([[np.cos(angle), np.sin(angle)], [np.sin(angle), np.cos(angle)]])

    move = 2.0 - 2.0
    t1_t2 = np.dot(v1_v2, rotate) + np.array([move, move * np.sqrt(3.0)])

    print np.shape(t1_t2)
    [X, Y] = t1_t2.T
    plt.scatter(X, Y, alpha=.5)
    plt.show()

    
    with open("uniform.txt", 'w') as fout:
        for point in t1_t2:
            fout.write(str(offset) + ' ' + str(offset+point[0]) + ' ' + str(offset+point[1]) + '\n')

if __name__ == '__main__':
    l()

