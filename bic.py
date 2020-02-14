import numpy as np
import matplotlib.pyplot as plt

def model1(x, params):
    y = params[0]+ x*params[1]+params[2]*x**2
    return y

def model2(x,params):
    y=params[0]*(np.exp(-0.5*(x-paramas[1])**2/params[2]**2))
    return y

def model3(x, params):
    y=params[0]*(np.exp(-0.5*(x-paramas[1])**2/params[2]**2))
    y+=params[0]*(np.exp(-0.5*(x-paramas[3])**2/params[4]**2))


    
    
def loglike(x_obs, y_obs, sigma_y_obs, params):
    n_obs = len(y_obs)
    l = 0.0
    for i in range(n_obs):
        l += -0.5*(y_obs[i]-model(x_obs[i,:], params))**2/sigma_y_obs[i]**2
    return l


def BIC_2(L_opti,k,n):
    BIC_2=(0.5)*(L_opti+(k/2)*np.log(n))
    return BIC


def probMdadoD(BIC,p):
    prob=exp(-BIC)*p
    return prob


def run_mcmc(data_file="data_to_fit.txt", n_dim=1, n_iterations=20000, sigma_y=1):
    data = np.loadtxt(data_file)
    x_obs = data[:,:n_dim]
    y_obs = data[:, n_dim]
    sigma_y_obs = data[:, n_dim+1]*sigma_y
    
    n_dim1=2
    n_dim2=2
    n_dim3=4

    params1 = np.zeros([n_iterations, n_dim1+1])
    params2=np.zeros([n_iterations, n_dim2+1])
    params3=np.zeros([n_iterations, n_dim3+1])
    
    for i in range(1, n_iterations):
        current_params1 = params1[i-1,:]
        next_params1 = current_params1 + np.random.normal(scale=0.01, size=n_dim+1)
        
        current_params2 = params2[i-1,:]
        next_params2 = current_params2 + np.random.normal(scale=0.01, size=n_dim+1)
        
        current_params3 = params3[i-1,:]
        next_params3 = current_params3 + np.random.normal(scale=0.01, size=n_dim+1)
        
        
        

        loglike_current1 = loglike(x_obs, y_obs, sigma_y_obs, current_betas1)
        loglike_next1 = loglike(x_obs, y_obs, sigma_y_obs, next_betas1)
        
        loglike_current2 = loglike(x_obs, y_obs, sigma_y_obs, current_betas2)
        loglike_next2 = loglike(x_obs, y_obs, sigma_y_obs, next_betas2)
        
        loglike_current3 = loglike(x_obs, y_obs, sigma_y_obs, current_betas3)
        loglike_next3 = loglike(x_obs, y_obs, sigma_y_obs, next_betas3)
        
        
        
        
        r1 = np.min([np.exp(loglike_next1 - loglike_current1), 1.0])
        alpha1 = np.random.random()
        
        r2 = np.min([np.exp(loglike_next2 - loglike_current2), 1.0])
        alpha2 = np.random.random()
        
        r3 = np.min([np.exp(loglike_next3 - loglike_current3), 1.0])
        alpha3 = np.random.random()


        
        if alpha1 < r1:
            params1[i,:] = next_params1
        else:
            params1[i,:] = current_params1
        
    
        if alpha2 < r2:
            params2[i,:] = next_params2
        else:
            params2[i,:] = current_params2
    
    
        if alpha3 < r3:
            params3[i,:] = next_params3
        else:
            params3[i,:] = current_params3
    
    
    params1 = params1[n_iterations//2:,:]
    params2 = params2[n_iterations//2:,:]
    params3 = params3[n_iterations//2:,:]
    
    return {'params1':params1, 'params2':params2, 'params3':params3, 'x_obs':x_obs, 'y_obs':y_obs}

n_dimplot1 =2 
results = run_mcmc()
params1 = results['params1']
params2 = results['params2']
params3 = results['params3']



plt.figure()
for i in range(0,n_dim+1):
    plt.subplot(2,3,i+1)
    plt.hist(params1[:,i],bins=15, density=True)
    plt.title(r"$\params1_{}= {:.2f}\pm {:.2f}$".format(i,np.mean(params1[:,i]), np.std(params1[:,i])))
    plt.xlabel(r"$\params1_{}$".format(i))
plt.subplots_adjust(hspace=0.5)
plt.savefig("ajuste_bayes_mcmc.png",  bbox_inches='tight')    
