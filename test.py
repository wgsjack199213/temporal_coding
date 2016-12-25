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
        if dirac_flag and t > t_input_dirac:    # The first time cross the t_input_dirac
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


def test_response():
    I_0 = 0.005
    theta_0 = -np.pi
    w = 0.1
    alpha = 0.6     # When alpha = 0.6, the cycle is about 40+ ms.


    #test_relation_alpha_cycle(I_0, theta_0, w)
    #return

    def pcr(I_0, theta_0, w, alpha):    # Compute Phase Response Curve
        line = []
        for t_dirac_input in np.linspace(0, 60):
            #print t_dirac_input
            t_fire, thetas = model(I_0, alpha, theta_0, w, t_dirac_input)
            t_fire_wo, thetas = model(I_0, alpha, theta_0, w, -1)

            #plt.plot(thetas)

            #print t_fire
            #print t_fire - t_input
            #print t_fire - t_fire_wo
            line.append(t_fire - t_fire_wo)
            label=str(w)
        return line, label

    for w in [0.1, 0.01, -0.1, -0.01]:
        line, label = pcr(I_0, theta_0, w, alpha)
        plt.plot(line, label=label)
    plt.legend()
    plt.savefig('Response properties of the theta model (I_0>0).png')
    plt.show()



if __name__ == '__main__':
    test_response()
