import numpy as np
import matplotlib.pyplot as plt


def theta_der(theta, alpha, I):
    return 1 - np.cos(theta) + alpha * I * (1 + np.cos(theta))

def model(I_0, alpha, theta_0, w, t_input_dirac):
    I = I_0

    theta = theta_0
    thetas = []
    dirac_flag = True
    for t in range(2000):
        thetas.append(theta)
        I = I_0
        epsilon = -0.00001
        if dirac_flag and t_input_dirac > epsilon and t > t_input_dirac:    # The first time cross the t_input_dirac
            I += w  # Synaptic currents as Diracs
            dirac_flag = False
        delta_theta = theta_der(theta, alpha, I)
        theta_new = theta + delta_theta
        #print theta_new
        
        if theta < np.pi < theta_new:  # cross pi
            return t, thetas
            #theta = theta_new
        else:
            theta = theta_new

    return t, thetas


def test_relation_alpha_cycle(I_0, theta_0, w):     
    '''
    Try various alpha to see the cycle of (or the time interval between) the neuron spikes
    '''
    X = []
    Y = []
    for alpha in [0.001, 0.005, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        t_fire_wo, thetas = model(I_0, alpha, theta_0, w, -1)
        X.append(alpha)
        Y.append(t_fire_wo)

    plt.scatter(X, Y)
    plt.show()


def pcr(I_0, theta_0, w, alpha):    # Compute Phase Response Curve
    line = []

    t_fire_wo, thetas = model(I_0, alpha, theta_0, w, -1)   # t_fire_wo means the firing time without any input transient synaptic current
    X, Y = [], []
    for t_dirac_input in np.linspace(0, 60, 300):
        #print t_dirac_input
        t_fire, thetas = model(I_0, alpha, theta_0, w, t_dirac_input)
        #plt.plot(thetas)

        Y.append(- t_fire + t_fire_wo)
        X.append(t_dirac_input)
        label=str(w)
    #print len(line)
    return X, Y, label


def test_response():
    I_0 = 0.005
    theta_0 = -np.pi
    w = 0.1
    alpha = 1.0     # When alpha = 1.0, the cycle is about 40+ ms.


    #test_relation_alpha_cycle(I_0, theta_0, w)
    #return


    # Figure 2 Left
    for w in [0.1, 0.01, -0.1, -0.01]:
        X, Y, label = pcr(I_0, theta_0, w, alpha)
        plt.plot(X, Y, label='w='+label)
    plt.legend()
    #plt.savefig('Response properties of the theta model (I_0>0).png')
    plt.show()

    # Figure 2 Right
    I_0 = -0.005
    theta_0 = np.arccos((1 + alpha * I_0) / (1 - alpha * I_0)) + 0.0001
    for w in [0.1, 0.001, -0.001]:
        X, Y, label = pcr(I_0, theta_0, w, alpha)
        plt.plot(X, Y, label='w='+label)
    plt.legend()
    plt.ylim(-60, 60)
    #plt.savefig('Response properties of the theta model (I_0<0).png')
    plt.show()


def test_network():
    I_0 = -0.01
    w = 0.1
    alpha = 0.1     # When alpha = 1.0, the cycle is about 40+ ms.
    theta_0 = -np.arccos((1 + alpha * I_0) / (1 - alpha * I_0)) 

    # Figure 2 Left
    for w in [0.5, 0.75, 1.0, 1.25, 1.5]:
        X, Y, label = pcr(I_0, theta_0, w, alpha)
        plt.plot(X, Y, label='w='+label)
    plt.legend()
    #plt.savefig('Response properties of the theta model (I_0>0).png')
    plt.show()


if __name__ == '__main__':
    test_response()

    #test_network()
