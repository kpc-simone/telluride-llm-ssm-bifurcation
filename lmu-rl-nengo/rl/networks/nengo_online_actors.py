import nengo
from nengo.network import Network
import numpy as np
import scipy.linalg
from scipy.special import legendre, eval_legendre, eval_sh_legendre, lpn
import scipy.integrate as integrate

from ..learning_rules import DPES
from nengo.learning_rules import PES

from .lmu_networks import LMUProcess, LMUNetwork


class Actor(nengo.Network):
    def __init__(self,n_neurons_state, n_neurons_pi, theta, d_state, d_action, discount, q_n, q_a, q_pi,
                  grad_clip=1,  learning_rate=1e-4, T_test=10000,state_ensembles=None,tau=0.05, prior=None,
                  **kwargs):
        super().__init__()
        
        self.activity_lmu_transform, self.action_lmu_transform, self.policy_lmu_transform = get_actor_transforms(algor, discount, n_neurons_state, theta, d_action,
                                                     q_n=q_n, q_a=q_a, q_pi=q_pi)

        if prior is None:
            prior = np.ones(d_action)/np.sqrt(d_action)

        with self:
            self.td_error = nengo.Node(size_in=1)
            self.action_input = nengo.Node(size_in=d_action)
            lmu_a = LMUProcess(theta=theta, q=q_a,size_in=d_action)
            self.action_memory = nengo.Node(lmu_a)
            nengo.Connection(self.action_input, self.action_memory,synapse=0)
            
            
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


            self.policy = nengo.Ensemble(n_neurons_pi, d_action)
            lmu_pi = LMUProcess(theta=theta, q=q_pi,size_in=d_action)
            self.policy_memory = nengo.Node(lmu_pi)
            nengo.Connection(self.policy, self.policy_memory, synapse=tau)
            
            self.learn_conn = nengo.Connection(self.state.neurons, self.policy, 
                                                transform=np.zeros((d_action,n_neurons_state)), 
                                        learning_rule_type = DPES(d_action,n_neurons_state, 1, 
                                                                learning_rate = learning_rate),synapse=tau)
            def get_error(t,x):
                phi_a = x[:d_action]
                mu_s = x[d_action:2*d_action]
                td = x[2*d_action]
                denom = np.sum(phi_a*mu_s)
                policy_grad = np.sign(denom)*td*phi_a/np.maximum(np.abs(denom), 1e-4)
                # policy_grad = 2*td*phi_a/np.maximum(denom, 1e-8)
                policy_grad = np.clip(policy_grad, -grad_clip, grad_clip)
                return np.hstack([policy_grad, x[(1+2*d_action):] ])
                
                
            self.error = nengo.Node(lambda t,x: get_error(t,x) if t<T_test else np.zeros(d_action + n_neurons_state), 
                               size_in = 1 + 2*d_action + n_neurons_state, size_out = d_action + n_neurons_state)
            
            nengo.Connection(self.action_input, self.error[:d_action], 
                             transform=self.action_lmu_transform, synapse=tau)
            nengo.Connection(self.policy_memory, self.error[d_action:2*d_action], 
                             transform=self.policy_lmu_transform, synapse=tau)
            nengo.Connection(self.td_error, self.error[2*d_action], synapse=None)
            
            nengo.Connection(self.state_memory, self.error[(1+2*d_action):], 
                             transform=self.activity_lmu_transform, synapse=None)
            
            nengo.Connection(self.error, self.learn_conn.learning_rule, synapse=None)
            self.rule = self.learn_conn.learning_rule

class Actor_NoMemory(nengo.Network):
    def __init__(self,n_neurons_state, n_neurons_pi, d_state, d_action, discount,  learning_rate=1e-4, 
                 T_test=10000,state_ensemble=None,tau=0.05, grad_clip=1, init_trans_scale=0, prior=None,eval_points=None,
                 policy_grad_pos=False,
                  state_kwargs={}, policy_kwargs={}):
        super().__init__()

        if prior is None:
            prior = np.ones(d_action)/np.sqrt(d_action)

        with self:
            self.td_error = nengo.Node(size_in=1)
            self.action_input = nengo.Node(size_in=d_action)
            if state_ensemble is not None:
                self.state = state_ensemble
            else:
                self.state_input = nengo.Node(size_in=d_state)
                self.state = nengo.Ensemble(n_neurons_state, d_state, **state_kwargs)
                nengo.Connection(self.state_input, self.state, synapse=None)
                

            self.policy = nengo.Ensemble(n_neurons_pi, d_action, **policy_kwargs)  
            self.learn_conn = nengo.Connection(self.state, self.policy, 
                                                function=lambda x: prior,eval_points=eval_points,#np.ones((d_action,n_neurons_state)), 
                                        learning_rule_type = PES(learning_rate = learning_rate),synapse=tau)
           
            if policy_grad_pos:
                def get_error(t,x):
                    phi_a = x[:d_action]
                    mu_s = x[d_action:2*d_action]
                    td = x[2*d_action]
                    denom = np.sum(phi_a*mu_s)
                    policy_grad = td*phi_a/np.maximum(np.abs(denom), 1e-4)
                    policy_grad = np.clip(policy_grad, -grad_clip, grad_clip)
                    return policy_grad
            else:
                def get_error(t,x):
                    phi_a = x[:d_action]
                    mu_s = x[d_action:2*d_action]
                    td = x[2*d_action]
                    denom = np.sum(phi_a*mu_s)
                    policy_grad = np.sign(denom)*td*phi_a/np.maximum(np.abs(denom), 1e-4)
                    # policy_grad = 2*td*phi_a/np.maximum(denom, 1e-8)
                    policy_grad = np.clip(policy_grad, -grad_clip, grad_clip)
                    return policy_grad
           
                
            self.error = nengo.Node(lambda t,x: get_error(t,x) if t<T_test else np.zeros(d_action), 
                               size_in = 1 + 2*d_action, size_out = d_action )
            
            nengo.Connection(self.action_input, self.error[:d_action],  synapse=tau)
            nengo.Connection(self.policy, self.error[d_action:2*d_action],  synapse=tau)
            nengo.Connection(self.td_error, self.error[2*d_action], synapse=None)
            
            nengo.Connection(self.error, self.learn_conn.learning_rule, synapse=None)
            self.rule = self.learn_conn.learning_rule

class QNetwork(nengo.Network):
    def __init__(self,n_neurons_state, n_neurons_q, theta, d_state, d_action, discount, q_n, q_a, q_r, q_q,
                 algor,  learning_rate=1e-4, T_test=10000,state_ensemble=None,tau=0.05, lambd=0.8,
                  **kwargs):
        super().__init__()
        
        self.activity_lmu_transform, self.action_lmu_transform, self.reward_lmu_transform, self.Q_transform, self.Q_lmu_transform = get_Q_transforms(algor, discount, n_neurons_state, theta, d_action,
                                                     q_n=q_n, q_a=q_a, q_r = q_r, q_v=q_q, lambd=lambd)


        with self:
            self.action_input = nengo.Node(size_in=d_action)
            lmu_a = LMUProcess(theta=theta, q=q_a,size_in=d_action)
            self.action_memory = nengo.Node(lmu_a)
            nengo.Connection(self.action_input, self.action_memory,synapse=0)
            
            self.state_input = nengo.Node(size_in=d_state)
            if state_ensemble is not None:
                self.state = state_ensemble
            else:
                self.state = nengo.Ensemble(n_neurons_state, d_state, **kwargs)
                nengo.Connection(self.state_input, self.state, synapse=None)
            lmu_s = LMUProcess(theta=theta, q=q_n, size_in=n_neurons_state)
            self.state_memory = nengo.Node(lmu_s)
            nengo.Connection(self.state.neurons, self.state_memory,synapse=tau)

            self.reward_input = nengo.Node(size_in=1)
            lmu_r = LMUProcess(theta=theta, q=q_r,size_in=1)
            self.reward_memory = nengo.Node(lmu_r)
            nengo.Connection(self.reward_input, self.reward_memory, synapse=None) 
    
            self.Q_vec = nengo.Ensemble(n_neurons_q,d_action)
            
            lmu_q = LMUProcess(theta=theta, q=q_q,size_in=1)
            self.Q = nengo.Node(lambda t,x: np.sum(x[:d_action]*x[d_action:]), size_in=2*d_action, size_out=1)
            nengo.Connection(self.Q_vec, self.Q[:d_action])
            nengo.Connection(self.action_input, self.Q[d_action:])
            self.Q_memory = nengo.Node(lmu_q)
            nengo.Connection(self.Q, self.Q_memory, synapse=tau)
            
            self.learn_connQ = nengo.Connection(self.state.neurons, self.Q_vec, 
                                                transform=np.zeros((d_action,n_neurons_state)), 
                                        learning_rule_type = DPES(d_action,n_neurons_state, 1, 
                                                                learning_rate = learning_rate),synapse=tau)

            self.error = nengo.Node(lambda t,x: np.hstack([x[0]*x[1:(1+d_action)], x[(1+d_action):]]) if t<T_test else np.zeros(d_action + n_neurons_state), 
                               size_in = 1 + d_action + n_neurons_state)
            
            nengo.Connection(self.reward_memory, self.error[0], 
                             transform=-self.reward_lmu_transform, synapse=None)
            nengo.Connection(self.Q, self.error[0], 
                             transform=-self.Q_transform, synapse=None)
            nengo.Connection(self.Q_memory, self.error[0],  
                             transform=self.Q_lmu_transform, synapse=None)
            
            nengo.Connection(self.action_memory, self.error[1:(1+d_action)],
                             transform=self.action_lmu_transform, synapse=None)
            
            nengo.Connection(self.state_memory, self.error[(1+d_action):], 
                             transform=self.activity_lmu_transform, synapse=None)
            
            nengo.Connection(self.error, self.learn_connQ.learning_rule, synapse=None)
            self.rule = self.learn_connQ.learning_rule




def get_actor_transforms(algor, discount, n_neurons, theta, size,
                              q_n=10, q_a=10, q_pi=10, lambd=0.8):
    activity_lmu_transform = np.asarray([legendre(i)(1) for i in range(q_n)])
    activity_lmu_transform = np.kron(np.eye(n_neurons), activity_lmu_transform.reshape(q_n, -1).T)

    action_lmu_transform = np.asarray([legendre(i)(1) for i in range(q_a)])
    action_lmu_transform = np.kron(np.eye(size), action_lmu_transform.reshape(q_a, -1).T)

    policy_lmu_transform = np.asarray([legendre(i)(1) for i in range(q_pi)])
    policy_lmu_transform = np.kron(np.eye(size), policy_lmu_transform.reshape(q_pi, -1).T)
    return activity_lmu_transform, action_lmu_transform, policy_lmu_transform



