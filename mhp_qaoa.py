import qiskit as qk
import numpy as np
from scipy.optimize import minimize
from itertools import product, combinations, permutations

def C3RXGate(theta):
    qc = qk.QuantumCircuit(4)
    qc.h(3)
    qc.p(theta / 8, [0, 1, 2, 3])
    qc.cx(0, 1)
    qc.p(-theta / 8, 1)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.p(-theta / 8, 2)
    qc.cx(0, 2)
    qc.p(theta / 8, 2)
    qc.cx(1, 2)
    qc.p(-theta / 8, 2)
    qc.cx(0, 2)
    qc.cx(2, 3)
    qc.p(-theta / 8, 3)
    qc.cx(1, 3)
    qc.p(theta / 8, 3)
    qc.cx(2, 3)
    qc.p(-theta / 8, 3)
    qc.cx(0, 3)
    qc.p(theta / 8, 3)
    qc.cx(2, 3)
    qc.p(-theta / 8, 3)
    qc.cx(1, 3)
    qc.p(theta / 8, 3)
    qc.cx(2, 3)
    qc.p(-theta / 8, 3)
    qc.cx(0, 3)
    qc.h(3)
    return qc.to_gate()

def Swap4Gate(theta):
    # TODO: This gate could probably be optimized further
    gate = qk.QuantumCircuit(4)
    gate.cx(2, 3)
    gate.cx(2, 0)
    gate.cx(1, 2)
    gate.x(2)
    gate.append(C3RXGate(theta=2*theta), (0, 2, 3, 1))
    gate.x(2)
    gate.cx(1, 2)
    gate.cx(2, 0)
    gate.cx(2, 3)
    return gate.to_gate()

def qubit_timestep_to_index(qubit, timestep, n):
    qubit = qubit % n
    timestep = timestep % n
    return qubit*n + timestep

def path_from_string(string, amount_nodes):
    path = [-1]*amount_nodes
    for i in range(amount_nodes):
        node_string = string[i*amount_nodes:i*amount_nodes+amount_nodes]
        node_position = node_string.find('1')
        path[node_position] = i
    return path

def compute_path_length(path, adj_matrix):
    length = 0
    for i,j in zip(path[:-1], path[1:]):
        length += adj_matrix[i,j]
    return length

def get_commutative_mapping(n_vertices):
    # TODO: Solve edge coloring problem
    # Return static solution for n_vertices <= 4
    if n_vertices == 3:
        p_col = (frozenset((frozenset((0,1)),)), frozenset((frozenset((0,2)),)),
                 frozenset((frozenset((1,2)),)))
        p_par = (frozenset((0,)), frozenset((1,)), frozenset((2,)))
        
        return list(product(p_par, p_col))
    elif n_vertices == 4:
        p_col = (frozenset((frozenset((0,1)), frozenset((2,3)))), frozenset((frozenset((0,2)), frozenset((1,3)))),
                 frozenset((frozenset((0,3)), frozenset((1,2)))))
        p_par = (frozenset((0,2)), frozenset((1,3)))
        
        return list(product(p_par, p_col))
    else:
        # TODO: Solve edge coloring problem
        raise Exception("Works only for n_vertices <= 4 in this version")
        
class MHP_QAOA:
    def __init__(self, n, p):
        # TODO: Implement get_commutative_mapping, then remove this assertion. Everything else
        # works for arbitrary TSP instance sizes
        assert (n <= 4), "Only works for problem size at most 4 in this version"
        
        self.circuit = None
        self.matrix_bound = None
        self.params_bound = None
        
        self.beta = None
        self.gamma = None
        self.matrix = None
        self.start_node = 0
        self.goal_node = 0
        self.core_nodes = None
        self.from_start = None
        self.to_goal = None
        
        self.path_lengths = None
        self.p = p
        self.n = n
        self.n_qubits = n**2
        self.mapping = get_commutative_mapping(n)

    def build_phase_separator(self, parameter):
        """
        Phase separator for a single iteration, hence only a single parameter is used
        """
        for i in range(self.n):
            id1 = qubit_timestep_to_index(i, 0, self.n)
            id2 = qubit_timestep_to_index(i, self.n-1, self.n)

            self.circuit.rz(2 * parameter * self.from_start[i], id1)
            self.circuit.rz(2 * parameter * self.to_goal[i], id2)
        
        for layer in self.mapping:
            # This should all happen in depth 1
            for timestep, qubits in product(layer[0], layer[1]):
                if timestep == self.n-1:
                    continue
                
                qs = tuple(qubits)
                
                id1 = qubit_timestep_to_index(qs[0], timestep, self.n)
                id2 = qubit_timestep_to_index(qs[1], timestep+1, self.n)
                self.circuit.rzz(2 * parameter * self.matrix[qs[0]][qs[1]], id1, id2)

                id1 = qubit_timestep_to_index(qs[1], timestep, self.n)
                id2 = qubit_timestep_to_index(qs[0], timestep+1, self.n)
                self.circuit.rzz(2 * parameter * self.matrix[qs[1]][qs[0]], id1, id2)

    def build_mixer(self, parameter):
        """
        Mixer for a single iteration, hence only a single parameter is used
        """
        def four_qubit_swap(u, v, t):
            i = t
            ip1 = t+1
            ui = qubit_timestep_to_index(u, i, self.n)
            uip1 = qubit_timestep_to_index(u, ip1, self.n)
            vi = qubit_timestep_to_index(v, i, self.n)
            vip1 = qubit_timestep_to_index(v, ip1, self.n)
            self.circuit.append(swap_gate, (ui, vi, uip1, vip1))

        swap_gate = Swap4Gate(parameter)
        for layer in self.mapping:
            # This should all happen in depth 1
            for timestep, qubits in product(layer[0], layer[1]):
                qs = tuple(qubits)
                four_qubit_swap(qs[0], qs[1], timestep)
                
    def compute_path_lengths(self, adj_matrix, start_node=0, goal_node=0):
        # This is kind of cheating but the Hamiltonian is too large to store
        self.path_lengths = {}
        for perm in permutations(range(self.n)):
            string = ""
            for i in range(self.n):
                k = perm.index(i)
                string += "0"*k + "1" + "0"*(self.n-k-1)
            path = [start_node] + [self.node_labels[i] for i in perm] + [goal_node]
            self.path_lengths[string] = compute_path_length(path, adj_matrix)

    def compute_expectation(self, counts, shots):
        sum_count = 0
        for string, count in counts.items():
            sum_count += self.path_lengths[string]*count

        return sum_count/shots

    def build_circuit(self, backend=None):
        self.beta = [qk.circuit.Parameter("beta{}".format(i)) for i in range(self.p)]
        self.gamma = [qk.circuit.Parameter("gamma{}".format(i)) for i in range(self.p)]
        self.matrix = [[qk.circuit.Parameter("matrix{0}{1}".format(i,j)) for j in range(self.n)] for i in range(self.n)]
        self.from_start = [qk.circuit.Parameter("fs{}".format(i)) for i in range(self.n)]
        self.to_goal = [qk.circuit.Parameter("tg{}".format(i)) for i in range(self.n)]

        self.circuit = qk.QuantumCircuit(self.n_qubits)
        
        # initial_state
        for i in range(0, self.n_qubits, self.n + 1):
            self.circuit.x(i)

        for i in range(self.p):
            self.build_phase_separator(self.gamma[i])
            self.build_mixer(self.beta[i])

        self.circuit.measure_all()
        self.circuit = qk.transpile(self.circuit, optimization_level=3, backend=backend)
    
    def bind_parameters(self, parameters):
        assert (self.matrix_bound is not None), "Matrix parameters need to be bound"
        betas = parameters[:self.p]
        gammas = parameters[self.p:]
        
        params_beta = {qcbeta: pbeta for qcbeta, pbeta in zip(self.beta, betas)}
        params_gamma = {qcgamma: pgamma for qcgamma, pgamma in zip(self.gamma, gammas)}
        
        self.params_bound = self.matrix_bound.assign_parameters({**params_beta, **params_gamma})
    
    def run_circuit(self, parameters, backend, shots=1000):
        self.bind_parameters(parameters)
        counts = backend.run(self.params_bound, seed_simulator=42, nshots=shots).result().get_counts()
        return counts
    
    def bind_matrix(self, adj_matrix, start_node=0, goal_node=0):
        self.core_nodes = sorted(frozenset(range(len(adj_matrix))).difference((start_node, goal_node)))
        self.node_labels = {i:k for i,k in enumerate(self.core_nodes)}
        params_matrix = {self.matrix[i][j]: adj_matrix[i,j] for i, j in product(range(self.n), range(self.n)) if i != j}
        params_from_start = {self.from_start[i]: adj_matrix[start_node, i] for i in range(self.n)}
        params_to_goal = {self.to_goal[i]: adj_matrix[i, goal_node] for i in range(self.n)}
        self.matrix_bound = self.circuit.assign_parameters({**params_matrix, **params_from_start, **params_to_goal})
        self.compute_path_lengths(adj_matrix, start_node, goal_node)
    
    def solve(self, adj_matrix, p, start_node=0, goal_node=0, shots=1000, backend=None):
        core_nodes = sorted(frozenset(range(len(adj_matrix))).difference((start_node, goal_node)))
        
        assert (self.n == len(core_nodes)), "Adjacency matrix does not fit with given number of qubits"
        
        if self.circuit is None or self.p != p:
            self.p = p
            self.build_circuit(backend)

        self.bind_matrix(adj_matrix, start_node, goal_node)
        
        if backend is None:
            backend = qk.Aer.get_backend('qasm_simulator')
        
        def get_circuit_expectation(parameters, evaluate=False):
            counts = self.run_circuit(parameters, backend, shots)
            expectation = self.compute_expectation(counts, shots)
            return expectation
        
        res = minimize(get_circuit_expectation, [1.0]*(self.p*2), method='COBYLA')
        optim = res.x
    
        counts = self.run_circuit(optim, backend)
        best_path = max(counts, key=counts.get)
        best_path = path_from_string(best_path, self.n)
        best_path = [start_node] + [self.node_labels[i] for i in best_path] + [goal_node]
        path_length = compute_path_length(best_path, adj_matrix)
    
        return tuple(best_path[1:]), path_length
