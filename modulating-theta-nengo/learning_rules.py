import nengo
import numpy as np
from nengo.network import Network
import warnings
import scipy

from nengo.builder import Builder, Signal
from nengo.builder.connection import get_eval_points, solve_for_decoders
from nengo.builder.operator import (
    DotInc, ElementwiseInc, Operator, Reset, SimPyFunc)
from nengo.exceptions import ValidationError
from nengo.learning_rules import LearningRuleType
from nengo.params import EnumParam, FunctionParam, NumberParam
from nengo.synapses import Lowpass, SynapseParam
from nengo.params import Default
from nengo.ensemble import Ensemble, Neurons
from nengo.builder.connection import slice_signal
from nengo.node import Node
from nengo.exceptions import BuildError

#####################################
#Synaptic Modulation rule: multiples decoders by a modulator signal
########################################

class SynapticModulation(LearningRuleType):
    modifies = "decoders"
    probeable = ("modulation")

    pre_synapse = SynapseParam("pre_synapse", default=Lowpass(tau=0.005), readonly=True)

    def __init__(self,  pre_synapse=Default):
        super(SynapticModulation, self).__init__(learning_rate=0,size_in=1)
        self.pre_synapse = pre_synapse

class SimSynapticModulation(Operator):
    def __init__(self, modulation, delta, weights, tag=None):
        super().__init__(tag=tag)
        self.id_weights = weights
        self.reads = [modulation, weights]
        self.updates = [delta]
        self.sets = []
        self.incs = []
        
    @property
    def delta(self):
        return self.updates[0]
    
    @property
    def modulation(self):
        return self.reads[0]
    
    @property
    def weights(self):
        return self.reads[1]
    
    def make_step(self, signals, dt, rng):
        weights = signals[self.weights]
        delta = signals[self.delta]
        rate = signals[self.modulation]
        def step_simsynmod():
             delta[...] = rate*self.id_weights - weights
        return step_simsynmod



@Builder.register(SynapticModulation)
def build_synapticmodulation(model, synmod, rule):
    conn = rule.connection

    # Create input modulation signal
    modulation = Signal(shape=rule.size_in, name="SynapticModulation:modulation")
    model.add_op(Reset(modulation))
    model.sig[rule]["in"] = modulation  # mod connection will attach here
    model.add_op(SimSynapticModulation(modulation, model.sig[rule]["delta"], model.sig[conn]["weights"]))
    # expose these for probes
    model.sig[rule]["modulation"] = modulation


####


class CumulativeSynapticModulation(LearningRuleType):
    modifies = "decoders"
    probeable = ("modulation")

    pre_synapse = SynapseParam("pre_synapse", default=Lowpass(tau=0.005), readonly=True)

    def __init__(self,  pre_synapse=Default):
        super(CumulativeSynapticModulation, self).__init__(learning_rate=0,size_in=1)
        self.pre_synapse = pre_synapse

class SimCumulativeSynapticModulation(Operator):
    def __init__(self, modulation, delta, weights, tag=None):
        super().__init__(tag=tag)
        self.reads = [modulation, weights]
        self.updates = [delta]
        self.sets = []
        self.incs = []
        
    @property
    def delta(self):
        return self.updates[0]
    
    @property
    def modulation(self):
        return self.reads[0]
    
    @property
    def weights(self):
        return self.reads[1]
    
#     @property
#     def decoders(self):
#         return self.updates[0]

    def make_step(self, signals, dt, rng):
        weights = signals[self.weights]
        delta = signals[self.delta]
        rate = signals[self.modulation]
        def step_simsynmod():
             delta[...] = (rate-1)*weights
        return step_simsynmod



@Builder.register(CumulativeSynapticModulation)
def build_cumulativesynapticmodulation(model, synmod, rule):
    conn = rule.connection

    # Create input modulation signal
    modulation = Signal(shape=rule.size_in, name="CumulativeSynapticModulation:modulation")
    model.add_op(Reset(modulation))
    model.sig[rule]["in"] = modulation  # mod connection will attach here
    model.add_op(SimCumulativeSynapticModulation(modulation, model.sig[rule]["delta"], model.sig[conn]["weights"]))
    # expose these for probes
    model.sig[rule]["modulation"] = modulation