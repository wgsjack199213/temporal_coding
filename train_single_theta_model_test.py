import numpy as np
import matplotlib.pyplot as plt


def theta_der(theta, alpha, I):
    return 1 - np.cos(theta) + alpha * I * (1 + np.cos(theta))

def model(I_0, alpha, theta_0, w, t_input_dirac, timestep=0.05, label = ''):
    I = I_0

    theta = theta_0
    thetas = []
    dirac_flag = 1
    fire = None
    cos_theta_minus, cos_theta_plus = None, None
    num_iter = 400
    for t in range(num_iter):
        thetas.append(theta)
        I = I_0
        epsilon = -0.00001
        if dirac_flag == 1 and t_input_dirac > epsilon and t * timestep > t_input_dirac:    # The first time cross the t_input_dirac
            I += w  # Synaptic currents as Diracs
            dirac_flag = 2
            

        delta_theta = theta_der(theta, alpha, I)
        theta_new = theta + delta_theta

        if dirac_flag == 2:
            cos_theta_minus = np.cos(theta)
            cos_theta_plus = np.cos(theta_new)
            dirac_flag = 0
        
        if theta < np.pi < theta_new:  # cross pi
            #return t, thetas
            fire = t * timestep
            theta = theta_new
        else:
            theta = theta_new

    if fire == None:
        fire = 20

    X = np.linspace(0, num_iter * timestep, num_iter)
    #print len(X), len(thetas)
    plt.plot(X, thetas, label=str(t_input_dirac) + ' ' + str(label))

    return fire, cos_theta_minus, cos_theta_plus




if __name__ == '__main__':
    '''
    Set w to different values, and run this program to train the theta model to let it fire at t=20ms.
    '''

    I_0 = -0.01
    alpha = 0.1     # When alpha = 1.0, the cycle is about 40+ ms.
    theta_0 = - np.arccos((1 + alpha * I_0) / (1 - alpha * I_0))
    #w = 0.7
    w = 0.8     #FIXME Change the initial weight value, and then run the program

    
    t_input_dirac = 0.3
    '''
    for w in [0.5, 0.6, 0.633, 0.635, 0.64, 0.65, 0.7]:
        fire = model(I_0, alpha, theta_0, w, t_input_dirac, timestep=0.05, label=w)

    plt.axhline(np.pi, color='g', ls='--', label='Firing threshold')
    plt.xlim(0, 25)
    plt.legend()
    plt.show()
    '''

    C = 1000
    eta = 0.000001
    fire_target = 5.0
    for i in range(200000):
        plt.cla()
        fire, cos_theta_minus, cos_theta_plus = model(I_0, alpha, theta_0, w, t_input_dirac, timestep=0.05, label=w)
        der_ts_wi = -alpha * (1 + cos_theta_minus) / ((1 - cos_theta_plus) + alpha * I_0 * (1 + cos_theta_plus))
        delta_wi = -2 * eta * (fire - fire_target) * der_ts_wi
        if der_ts_wi <= -C or der_ts_wi >= 0:
            print '#!! der_ts_wi', der_ts_wi, 'fire:', fire
            delta_wi = 2 * eta * (fire - fire_target) * C
            print 'cos_theta_minus and plus:', cos_theta_minus, cos_theta_plus, 'delta_wi:', delta_wi

        if i % 1 == 0:
            print i,
            print 'w:', w, 'delta_wi:', delta_wi, 'fire:', fire, 'error:', fire_target - fire
        w += delta_wi
        #print fire_target
        if np.abs(fire_target - fire) < 0.05:
            break

    plt.axhline(np.pi, color='g', ls='--', label='Firing threshold')
    plt.xlim(0, 25)
    plt.legend()
    plt.show()


    
