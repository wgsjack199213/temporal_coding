import numpy as np
import matplotlib.pyplot as plt


def theta_der(theta, alpha, I):
    return 1 - np.cos(theta) + alpha * I * (1 + np.cos(theta))

def model(I_0, alpha, theta_0, w, t_input_dirac, timestep=0.05, label = '', ls='-'):
    I = I_0

    theta = theta_0
    thetas = []
    dirac_flag = 1
    fire = None
    cos_theta_minus, cos_theta_plus = None, None
    num_iter = 601
    for t in range(num_iter):
        thetas.append(theta)
        I = I_0
        epsilon = -0.00001
        if dirac_flag == 1 and t_input_dirac > epsilon and t * timestep > t_input_dirac:    # The first time cross the t_input_dirac
            I += w  # Synaptic currents as Diracs
            dirac_flag = 2
            

        delta_theta = theta_der(theta, alpha, I)
        if t % 10 == 0:
            print theta, np.cos(theta), delta_theta
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
    plt.plot(X, thetas, label=str(label), ls=ls)

    return fire, cos_theta_minus, cos_theta_plus




if __name__ == '__main__':
    '''
    Set w to different values, and run this program to train the theta model to let it fire at t=20ms.
    '''

    I_0 = -0.01
    alpha = 0.1     # When alpha = 1.0, the cycle is about 40+ ms.
    theta_0 = - np.arccos((1 + alpha * I_0) / (1 - alpha * I_0)) + 0.000001
    #w = 0.7
    w = 0.8     #FIXME Change the initial weight value, and then run the program

    '''
    t_input_dirac = 3.0
    for w in [-10.0, -1.0, 0.5, 0.6328, 0.6329, 0.633, 0.635, 0.7, 1.0]:
        if w < 0:
            ls = '--'
        else:
            ls = '-'
        fire = model(I_0, alpha, theta_0, w, t_input_dirac, timestep=0.05, label='w=' + str(w), ls=ls)

    plt.axhline(np.pi, color='orange', ls='--', label='Firing threshold')
    plt.xlim(0, 25)
    plt.legend(loc='lower right', fontsize=12)
    plt.xlabel('Time t (ms)', fontsize=14)
    plt.ylabel("Theta(t)", fontsize=14)
    plt.savefig("theta_t_negtive_I.png")
    plt.show()
    '''
    
    '''
    I_0 = 0.01
    t_input_dirac = 3.0
    w = 0.1
    
    for w in [-10.0, 0.0, 2.0, 5.0, 10.0]:
    #for t_input_dirac in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]:
        if w < 0:
            ls = '--'
        else:
            ls = '-'
        fire = model(I_0, alpha, theta_0, w, t_input_dirac, timestep=0.05, label='w=' + str(w), ls=ls)

    plt.axhline(np.pi, color='orange', ls='--', label='Firing threshold')
    plt.xlim(0, 15)
    plt.ylim(-3, 10)
    plt.legend(loc='upper right', fontsize=12)
    plt.xlabel('Time t (ms)', fontsize=14)
    plt.ylabel("Theta(t)", fontsize=14)
    plt.savefig("theta_t_positive_I.png")
    plt.show()
    '''


    '''
    I_0 = 0.01
    t_input_dirac = 3.0
    w = 0.1
    
    #for w in [-10.0, 0.0, 2.0, 5.0, 10.0]:
    for t_input_dirac in [0.0, 0.2, 0.5, 0.8, 1.0, 2.0, 3.0]:
        if w < 0:
            ls = '--'
        else:
            ls = '-'
        fire = model(I_0, alpha, theta_0, w, t_input_dirac, timestep=0.05, label='t_i=' + str(t_input_dirac), ls=ls)

    plt.axhline(np.pi, color='orange', ls='--', label='Firing threshold')
    plt.xlim(0, 15)
    plt.ylim(-3, 10)
    plt.legend(loc='upper right', fontsize=12)
    plt.xlabel('Time t (ms)', fontsize=14)
    plt.ylabel("Theta(t)", fontsize=14)
    #plt.savefig("theta_t_positive_I.png")
    plt.show()
    '''


    # pca check
    I_0 = -0.01
    t_input_dirac = 3.0
    w = 0.1
    theta_0 = - np.arccos((1 + alpha * I_0) / (1 - alpha * I_0))
    
    for w in [-10.0, 0.0, 2.0, 5.0, 10.0, 20.0, 30.0]:
    #for t_input_dirac in [0.0, 0.2, 0.5, 0.8, 1.0, 2.0, 3.0]:
        if w < 0:
            ls = '--'
        else:
            ls = '-'
        fire = model(I_0, alpha, theta_0, w, t_input_dirac, timestep=0.05, label='w=' + str(w), ls=ls)

    plt.axhline(np.pi, color='orange', ls='--', label='Firing threshold')
    plt.legend(loc='upper right', fontsize=12)
    plt.xlabel('Time t (ms)', fontsize=14)
    plt.ylabel("Theta(t)", fontsize=14)
    #plt.savefig("theta_t_positive_I.png")
    plt.show()
    

    



    
    C = 1000
    eta = 0.000001
    fire_target = 5.0
    for i in range(200000):
        break
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


    
