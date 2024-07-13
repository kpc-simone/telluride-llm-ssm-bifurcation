import nengo
from nengo.network import Network
import numpy as np
import scipy.linalg
from scipy.special import legendre, eval_legendre, eval_sh_legendre, lpn
import scipy.integrate as integrate

from ..learning_rules import DPES
from nengo.learning_rules import PES

from .lmu_networks import LMUProcess, LMUNetwork

##############################
# General network that learns a value function given LMUs (in nodes) & decoders
#######################

class ValueCritic(nengo.Network):
    def __init__(self,n_neurons_state, n_neurons_value, theta, d, discount, q_n, q_r, q_v,
                 algor,  learning_rate=1e-4, T_test=10000,state_ensembles=None,tau=0.05, lambd=0.8,
                  **kwargs):
        super().__init__()
        if "activity_lmu_transform" in kwargs:
            self.activity_lmu_transform = kwargs.pop("activity_lmu_transform")
            self.reward_lmu_transform = kwargs.pop("reward_lmu_transform")
            self.value_lmu_transform = kwargs.pop("value_lmu_transform")
        else:
            self.activity_lmu_transform, self.reward_lmu_transform, self.value_lmu_transform = get_critic_transforms(algor, discount, n_neurons_state, theta,
                                                     q_a=q_n, q_r = q_r, q_v=q_v, lambd=lambd)


        if algor=='TDLambda':
            pre_act_input_size=q_n
        else:
            pre_act_input_size=1
        with self:
            self.reset = nengo.Node(size_in=1)
            self.state_input = nengo.Node(size_in=d)
            if state_ensembles is None:
                self.state_input = nengo.Node(size_in=d)
                self.state = nengo.Ensemble(n_neurons_state, d, **kwargs)
                nengo.Connection(self.state_input, self.state, synapse=None)
                lmu_s = LMUProcess(theta=theta, q=q_n, size_in=n_neurons_state,with_resets=True)
                self.state_memory = nengo.Node(lmu_s)
                nengo.Connection(self.state.neurons, self.state_memory[1:],synapse=tau)
                nengo.Connection(self.reset, self.state_memory[0],synapse=None)
            elif len(state_ensembles)==1:
                self.state = state_ensembles[0]
                lmu_s = LMUProcess(theta=theta, q=q_n, size_in=n_neurons_state,with_resets=True)
                self.state_memory = nengo.Node(lmu_s)
                nengo.Connection(self.state.neurons, self.state_memory[1:],synapse=tau)
                nengo.Connection(self.reset, self.state_memory[0],synapse=None)
            elif len(state_ensembles)==2:
                self.state = state_ensembles[0]
                self.state_memory = state_ensembles[1]
                
                

            self.reward_input = nengo.Node(size_in=1)
            lmu_r = LMUProcess(theta=theta, q=q_r,size_in=1,with_resets=True)
            self.reward_memory = nengo.Node(lmu_r)
            nengo.Connection(self.reward_input, self.reward_memory[1:], synapse=None) 
            nengo.Connection(self.reset, self.reward_memory[0],synapse=None)

            lmu_v = LMUProcess(theta=theta, q=q_v,size_in=1,with_resets=True)
            self.value = nengo.Ensemble(n_neurons_value,1)
            self.value_memory = nengo.Node(lmu_v)
            nengo.Connection(self.value, self.value_memory[1:], synapse=tau)
            nengo.Connection(self.reset, self.value_memory[0],synapse=None)
            
            self.learn_connV = nengo.Connection(self.state.neurons, self.value, 
                                                transform=np.zeros((1,n_neurons_state)), 
                                        learning_rule_type = DPES(1,n_neurons_state, 
                                                                pre_act_input_size, 
                                                                learning_rate = learning_rate),
                                                synapse=tau)
            
            self.td_error = nengo.Node(size_in=pre_act_input_size)
            self.error = nengo.Node(lambda t,x: x if t<T_test else 0, 
                               size_in=(1 + n_neurons_state)*pre_act_input_size)
            
            nengo.Connection(self.reward_memory, self.td_error, 
                             transform=-self.reward_lmu_transform)
            nengo.Connection(self.value_memory, self.td_error,  
                             transform=self.value_lmu_transform, synapse=None)

            nengo.Connection(self.td_error, self.error[:pre_act_input_size], synapse=None)
            nengo.Connection(self.state_memory, self.error[pre_act_input_size:], 
                             transform=self.activity_lmu_transform, synapse=None)
            
            nengo.Connection(self.error, self.learn_connV.learning_rule, synapse=None)
            self.rule = self.learn_connV.learning_rule

        

#Function to return decoders need for LMU RL rules (TD0, TDtheta, TDlambda)
#( these are descibed below)

# The decoders returned are activity_lmu_transform, reward_lmu_transform, value_transform, value_lmu_transform
# 
# The DPES rule will recieve input pre_activity and error (in a single vector)
# The update to state decoders will be of the form,
#    pre_activity @ error.T
# pre_activity is computed as
#    activity_lmu_transform @ A
# where A is either decoded activities or an LMU representation of activites depending on the rule
# The error is computed with three terms,
#    value_lmu_transform @ V - value_transform @ v - reward_lmu_transform @ R
# where v is the current value of the function we are learning (value usually, but could be Q or SR)
# V is the LMU representation of that function
# The function we want to learn is a discounted sum/integral of some variable, usually reward
#  R is the LMU representation of the reward
# For value learning r is the reward
# For SR learning r is the env state



def get_critic_transforms(rule_type, discount, n_neurons, theta, size=1,
                    q_a=10, q_r = 10, q_v=10, alpha=10, lambd=0.8,
                          lambda_w_fun=lambda x,l,t: -np.log(l)*l**(x*t),
                         epsabs=1e-4):
    #lambda_w_fun=lambda x,l,t: -np.log(l)*l**(k*t)
    #lambda_w_fun=lambda x,l,t: np.exp(-l*x*t)
    w_fun = lambda x: lambda_w_fun(x,lambd,theta)
    
    if rule_type=="TD0":
        activity_lmu_transform = eval_sh_legendre(np.arange(q_a).reshape(1,-1), 0)
        activity_lmu_transform = np.kron(np.eye(n_neurons), activity_lmu_transform)

        reward_lmu_transform = eval_sh_legendre(np.arange(q_r).reshape(1,-1), 0 )
        
        legs = lpn(q_v-1, 0) 
        value_lmu_transform = (np.log(discount)*legs[0] - (1/theta)*legs[1]).reshape(1,-1)
    elif rule_type=="TD0euler":
        activity_lmu_transform = eval_sh_legendre(np.arange(q_a).reshape(1,-1), 1) 
        activity_lmu_transform = np.kron(np.eye(n_neurons), activity_lmu_transform.reshape(q_a, -1).T)
    
        reward_lmu_transform = eval_sh_legendre(np.arange(q_r).reshape(1,-1), 1) 
        reward_lmu_transform = np.kron(np.eye(size), reward_lmu_transform.reshape(q_r, -1).T)
        
        dt = 0.1*theta
        mat1 = eval_sh_legendre(np.arange(q_v).reshape(1,-1), 1) 
        mat1 =np.kron(np.eye(size), mat1.reshape(q_v, -1).T)
        mat2 = eval_sh_legendre(np.arange(q_v).reshape(1,-1), 0.9) 
        mat2 =np.kron(np.eye(size), mat2.reshape(q_v, -1).T)
        value_lmu_transform = (np.log(discount) - 1/dt)*mat1 + mat2/dt

        
    elif rule_type=="TDtheta":
        reward_lmu_transform = np.zeros(q_r)
        for i in range(q_r):
            intgrand = lambda x: (1/theta)*(discount**(theta*(1-x)))*legendre(i)(2*x-1)
            reward_lmu_transform[i]= integrate.quad(intgrand, 0,1,epsabs=epsabs)[0]
    
        reward_lmu_transform =  np.kron(np.eye(size), reward_lmu_transform.reshape(1, -1))
        
        activity_lmu_transform =  eval_sh_legendre(np.arange(q_a).reshape(1,-1), 1) 
        activity_lmu_transform = np.kron(np.eye(n_neurons), activity_lmu_transform.reshape(q_a, -1).T)

        value_lmu_transform =  eval_sh_legendre(np.arange(q_v).reshape(1,-1), 1) -   discount**theta * eval_sh_legendre(np.arange(q_v).reshape(1,-1), 0) 
        value_lmu_transform = np.kron(np.eye(size), value_lmu_transform.reshape(q_v, -1).T)

    elif rule_type=="TDlambda":
        reward_lmu_transform = np.zeros(q_r)
        for i in range(q_r):
            intgrand = lambda y,x: (1/theta)*w_fun(x)*(discount**(theta*(1-y)))*legendre(i)(2*y-1)
            reward_lmu_transform[i]=integrate.dblquad(intgrand, 0,1,lambda x: x, lambda x: 1, epsabs=epsabs)[0]
        reward_lmu_transform =  np.kron(np.eye(size), reward_lmu_transform.reshape(1, -1))
        
        activity_lmu_transform =  eval_sh_legendre(np.arange(q_a).reshape(1,-1), 1)
        activity_lmu_transform = np.kron(np.eye(n_neurons), activity_lmu_transform.reshape(q_a, -1).T)


        value_lmu_transform1 = np.zeros(q_v)
        for i in range(q_v):
            intgrand = lambda x: w_fun(x)*(discount**(theta*x))*legendre(i)(2*x-1)
            value_lmu_transform1[i]=integrate.quad(intgrand, 0, 1,epsabs=epsabs)[0]
        
        value_lmu_transform2 =  eval_sh_legendre(np.arange(q_v).reshape(1,-1), 1)
        value_lmu_transform = np.kron(np.eye(size), (value_lmu_transform1-value_lmu_transform2).reshape(q_v, -1).T)

    elif rule_type=="TDLambda":
        activity_lmu_transform = 1
        reward_lmu_transform = np.zeros((q_r, q_a))
        value_lmu_transform = np.zeros((q_v, q_a))
        
        for i in range(q_a):
            for j in range(q_r):
                integrand = lambda y,x: (discount**(x-y)) * legendre(i)(2*(x/theta)-1) * legendre(j)(2*(y/theta)-1)
                reward_lmu_transform[j,i]=integrate.dblquad(integrand, 0,theta,lambda x: 0, lambda x: x, epsabs=epsabs)[0]
            for j in range(q_v):
                integrand = lambda x: ( ((discount**x) * legendre(j)(0)) - legendre(j)(2*(x/theta)-1)) * legendre(i)(2*(x/theta)-1) 
                value_lmu_transform[j,i]=integrate.quad(integrand, 0,theta, epsabs=epsabs)[0]

        value_lmu_transform = np.kron(np.eye(size), value_lmu_transform.T)
        reward_lmu_transform = np.kron(np.eye(size), reward_lmu_transform.T)
    # elif rule_type=="TDLambda":
    #     activity_lmu_transform = 1
    #     reward_lmu_transform = np.zeros((q_r, q_a))
    #     value_transform = np.zeros((1,q_a))
    #     value_lmu_transform = np.zeros((q_v, q_a))
        
        
    #     for i in range(q_a):
    #         intgrand = lambda x: (discount**(x*theta))*legendre(i)(2*x-1)
    #         value_transform[0,i]=(1/theta)*integrate.quad(intgrand, 0,1,epsabs=epsabs)[0]

    #         for j in range(q_r):
    #             intgrand = lambda y,x: (1/(y + 1e-8))*(discount**(theta*y*(1-x)))*legendre(i)(2*x-1)*legendre(j)(2*y-1)
    #             reward_lmu_transform[j,i]=(1/theta)*integrate.dblquad(intgrand, 0,1,0,1, epsabs=epsabs)[0]

    #     for i in range(np.min([q_v,q_a])):
    #         intgrand = lambda x: legendre(i)(2*x-1)**2
    #         value_lmu_transform[i,i]=(1/theta)*integrate.quad(intgrand, 0, 1, epsabs=epsabs)[0]
            
    #     value_transform =   eval_sh_legendre(np.arange(q_v)[:,None], 1) @ value_transform
    #     value_lmu_transform = value_lmu_transform - value_transform
    #     value_lmu_transform = np.kron(np.eye(size), value_lmu_transform.T)
    #     reward_lmu_transform = np.kron(np.eye(size), reward_lmu_transform.T)

    else:
        print("Not a valid rule")
    return activity_lmu_transform, reward_lmu_transform, value_lmu_transform 


# def get_critic_transforms2(rule_type, discount, n_neurons, theta, size=1,
#                     q_a=10, q_r = 10, q_v=10, alpha=10, lambd=0.8,
#                           lambda_w_fun=lambda x,l,t: -np.log(l)*l**(x*t),
#                          epsabs=1e-4):
#     #lambda_w_fun=lambda x,l,t: -np.log(l)*l**(k*t)
#     #lambda_w_fun=lambda x,l,t: np.exp(-l*x*t)
#     w_fun = lambda x: lambda_w_fun(x,lambd,theta)
    
#     if rule_type=="TD0":
#         activity_lmu_transform = eval_sh_legendre(np.arange(q_a).reshape(1,-1), 0)
#         activity_lmu_transform = np.kron(np.eye(n_neurons), activity_lmu_transform)

#         reward_lmu_transform = eval_sh_legendre(np.arange(q_r).reshape(1,-1), 0 )
        
#         legs = lpn(q_v-1, 0) 
#         value_lmu_transform = (np.log(discount)*legs[0] - (1/theta)*legs[1]).reshape(1,-1)
#     elif rule_type=="TD0euler":
#         activity_lmu_transform = eval_sh_legendre(np.arange(q_a).reshape(1,-1), 1) 
#         activity_lmu_transform = np.kron(np.eye(n_neurons), activity_lmu_transform.reshape(q_a, -1).T)
    
#         reward_lmu_transform = eval_sh_legendre(np.arange(q_r).reshape(1,-1), 1) 
#         reward_lmu_transform = np.kron(np.eye(size), reward_lmu_transform.reshape(q_r, -1).T)
        
#         dt = 0.1*theta
#         mat1 = eval_sh_legendre(np.arange(q_v).reshape(1,-1), 1) 
#         mat1 =np.kron(np.eye(size), mat1.reshape(q_v, -1).T)
#         mat2 = eval_sh_legendre(np.arange(q_v).reshape(1,-1), 0.9) 
#         mat2 =np.kron(np.eye(size), mat2.reshape(q_v, -1).T)
#         value_lmu_transform = (np.log(discount) - 1/dt)*mat1 + mat2/dt

        
#     elif rule_type=="TDtheta":
#         reward_lmu_transform = np.zeros(q_r)
#         for i in range(q_r):
#             intgrand = lambda x: (1/theta)*(discount**(theta*(1-x)))*legendre(i)(2*x-1)
#             reward_lmu_transform[i]= np.sum(intgrand(np.linspace(0,1,100)))/100
    
#         reward_lmu_transform =  np.kron(np.eye(size), reward_lmu_transform.reshape(1, -1))
        
#         activity_lmu_transform =  eval_sh_legendre(np.arange(q_a).reshape(1,-1), 1) 
#         activity_lmu_transform = np.kron(np.eye(n_neurons), activity_lmu_transform.reshape(q_a, -1).T)

#         value_lmu_transform =  eval_sh_legendre(np.arange(q_v).reshape(1,-1), 1) -   discount**theta * eval_sh_legendre(np.arange(q_v).reshape(1,-1), 0) 
#         value_lmu_transform = np.kron(np.eye(size), value_lmu_transform.reshape(q_v, -1).T)

#     elif rule_type=="TDlambda":
#         reward_lmu_transform = np.zeros(q_r)
#         for i in range(q_r):
#             intgrand = lambda y,x: (1/theta)*w_fun(x)*(discount**(theta*(1-y)))*legendre(i)(2*y-1)
#             reward_lmu_transform[i]=integrate.dblquad(intgrand, 0,1,lambda x: x, lambda x: 1, epsabs=epsabs)[0]
#         reward_lmu_transform =  np.kron(np.eye(size), reward_lmu_transform.reshape(1, -1))
        
#         activity_lmu_transform =  eval_sh_legendre(np.arange(q_a).reshape(1,-1), 1)
#         activity_lmu_transform = np.kron(np.eye(n_neurons), activity_lmu_transform.reshape(q_a, -1).T)


#         value_lmu_transform1 = np.zeros(q_v)
#         for i in range(q_v):
#             intgrand = lambda x: w_fun(x)*(discount**(theta*x))*legendre(i)(2*x-1)
#             value_lmu_transform1[i]=np.sum(intgrand(np.linspace(0,1,100)))/100
        
#         value_lmu_transform2 =  eval_sh_legendre(np.arange(q_v).reshape(1,-1), 1)
#         value_lmu_transform = np.kron(np.eye(size), (value_lmu_transform1-value_lmu_transform2).reshape(q_v, -1).T)

        
#     elif rule_type=="TDLambda":
#         activity_lmu_transform = 1
#         reward_lmu_transform = np.zeros((q_r, q_a))
#         value_transform = np.zeros((1,q_a))
#         value_lmu_transform = np.zeros((q_v, q_a))
        
        
#         for i in range(q_a):
#             intgrand = lambda x: (discount**(x*theta))*legendre(i)(2*x-1)
#             value_transform[0,i]= (1/theta)*np.sum(intgrand(np.linspace(0,1,100)))/100 

#             for j in range(q_r):
#                 intgrand = lambda y,x: (1/(y + 1e-8))*(discount**(theta*y*(1-x)))*legendre(i)(2*x-1)*legendre(j)(2*y-1)
#                 ptss = np.vstack(np.meshgrid(np.linspace(0,1,100),np.linspace(0,1,100)).T
#                 reward_lmu_transform[j,i]=(1/theta)*np.sum(intgrand(ptss[:,0],ptss[:,1]) )/ (100**2)

#         for i in range(np.min([q_v,q_a])):
#             intgrand = lambda x: legendre(i)(2*x-1)**2
#             value_lmu_transform[i,i]=(1/theta)*np.sum(intgrand(np.linspace(0,1,100)))/100 
            
#         value_transform =   eval_sh_legendre(np.arange(q_v)[:,None], 1) @ value_transform
#         value_lmu_transform = value_lmu_transform - value_transform
#         value_lmu_transform = np.kron(np.eye(size), value_lmu_transform.T)
#         reward_lmu_transform = np.kron(np.eye(size), reward_lmu_transform.T)

#     else:
#         print("Not a valid rule")
#     return activity_lmu_transform, reward_lmu_transform, value_lmu_transform 
