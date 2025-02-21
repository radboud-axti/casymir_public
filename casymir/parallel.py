"""
casymir.parallel
~~~~~~~~~~~~~~~~~~

This module contains all the elements pertaining to the parallel path implementation.
This implementation follows the theoretical framework described in:

-   J. Yao and I.A. Cunningham "Parallel Cascades: New ways to describe noise transfer in medical imaging systems."
    in Medical Physics 28, 2001. doi: 10.1118/1.1405842
-   M. Sattarivand and I.A. Cunningham "Computational Engine for Development of Complex Cascaded Models of Signal and
    Noise in X-Ray Imaging Systems." in IEEE Transcactions on Medical Imaging, Vol 24, No. 2, 2005.
    doi: 10.1109/TMI.2004.839680

Classes:

- Node: Represents a branching point in the model (Bernoulli split or Cascade Fork).

- Path: Represents a sequence of processes applied to a Signal.

Functions:

- extract_paths: Extracts paths and their probabilities from a branching structure.

- apply_parallel_process: Applies a parallel processing structure, including cross-spectral term computation.

Auxilliary Functions:

- find_lowest_common_ancestor: Identifies the lowest common ancestor (LCA) of two Nodes.

- get_lca_probability: Retrieves the probability assigned to a Path the LCA Node.

- find_prob_before_lca: Determines the probability of a given Path before reaching the LCA Node, used
                        for adequately scaling the input Wiener Spectra and Mean Quanta for the cross-term computation.
"""
import numpy as np
import copy


class Node:
    """
    Represents a branching point in the model.

    - "Bernoulli" Nodes split into two mutually exclusive paths.
    - "Fork" Nodes deterministically send quanta into multiple parallel processes.
    """

    def __init__(self, node_type, parent=None, name=None):
        """
        Initialize a Node.

        :param node_type: Either "Bernoulli" (mutually exclusive) or "Fork" (deterministic).
        :param parent: The parent node, if applicable.
        """
        if node_type not in ["Bernoulli", "Fork"]:
            raise ValueError("Node type must be 'Bernoulli' or 'Fork'.")
        if name is None:
            name = node_type

        self.node_type = node_type
        self.name = name
        self.parent = parent
        self.branches = []

    def add(self, branch, probability=1.0):
        """
        Add a path or a sub-node to this node.

        :param branch: Either a Path object or another Node object.
        :param probability: The probability of taking this branch (only relevant for Bernoulli nodes).
        """
        if self.node_type == "Bernoulli" and len(self.branches) >= 2:
            raise ValueError("Bernoulli nodes can only have two branches.")

        self.branches.append((branch, probability))
        if isinstance(branch, Node):
            branch.parent = self


class Path:
    """
    Represents a single path in a parallel cascade.
    A path is a sequence of processes applied to a signal.
    """

    def __init__(self, processes, name="Path"):
        """
        Initialize a Path.

        :param processes: List of process functions to apply in sequence. These must correspond to a pre-defined process
        with arguments (detector, spectrum, signal).
        """
        self.processes = processes
        self.name = name

    def apply(self, signal, detector, spectrum):
        """
        Apply all processes in this path sequentially, accumulating gains and transfer functions.

        :param signal: CASYMIR Signal object.
        :param detector: CASYMIR Detector object.
        :param spectrum: CASYMIR Spectrum object.
        :return: Modified signal, cumulative gain, and cumulative spread function.
        """

        cumulative_gain = 1.0
        cumulative_transfer = np.ones_like(signal.freq)

        for process in self.processes:
            signal, g, t = process(detector, spectrum, signal)
            cumulative_gain *= g
            cumulative_transfer *= t

        return signal, cumulative_gain, cumulative_transfer


def extract_paths(node, probability_list=None, parent=None):
    """
    Extracts all the Paths belonging to the specified Node object and tracks the probability history.

    :param node: Node object
    :param probability_list: List of probabilities along this path.
    :param parent: The parent Node of the current branch.
    :return: A list of (path, probability history, parent) tuples.
    """
    if probability_list is None:
        probability_list = [1.0]

    paths_with_probs = []

    for branch, branch_prob in node.branches:
        new_prob_list = probability_list + [branch_prob]

        if isinstance(branch, Path):
            paths_with_probs.append((branch, new_prob_list, node))
        elif isinstance(branch, Node):
            paths_with_probs.extend(extract_paths(branch, new_prob_list, node))

    return paths_with_probs


def apply_parallel_process(parent_node, signal, detector, spectrum):
    """
    Applies a parallel process structure (branching nodes and paths) to a signal,
    taking into account cross-spectral terms.

    :param parent_node: The root Node object representing the parallel process.
    :param signal: The CASYMIR Signal object being modified.
    :param detector: The CASYMIR Detector object.
    :param spectrum: The CASYMIR Spectrum object.
    :return: The modified signal after applying all branches, including cross-spectral terms.
    """

    print("Parallel path combination begins here")
    print("-------------------------------------")

    # Extract paths, probability history, and parent nodes
    paths = extract_paths(parent_node)
    path_signals = []
    path_gains = []
    path_transfers = []
    path_probs = []
    path_parents = []
    path_prob_histories = []

    for path, prob_history, parent in paths:
        cumulative_prob = np.prod(prob_history)
        sig_out, g_out, t_out = path.apply(signal, detector, spectrum)
        path_signals.append(sig_out)
        path_gains.append(g_out)
        path_transfers.append(t_out)
        path_probs.append(cumulative_prob)
        path_parents.append(parent)
        path_prob_histories.append(prob_history)
        print("{}: {}, Probability History: {}, Cumulative Prob: {}".format(path.name, path.processes, prob_history, cumulative_prob))

    # Initialize combined Signal
    combined_signal = copy.copy(signal)
    combined_signal.mean_quanta = 0
    combined_signal.signal = 0
    combined_signal.wiener = 0

    # Sum contributions from each path
    for i, sig in enumerate(path_signals):
        combined_signal.mean_quanta += sig.mean_quanta * path_probs[i]
        combined_signal.signal += sig.signal * path_probs[i]
        combined_signal.wiener += sig.wiener * path_probs[i]

    # Pair-wise computation of cross-spectral terms
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            W_ab = 0
            path_a, prob_history_a, parent_a = paths[i]
            path_b, prob_history_b, parent_b = paths[j]
            # Find the lowest common ancestor, that is, the shared Node from which the Paths initially stem.
            common_ancestor = find_lowest_common_ancestor(parent_a, parent_b)

            if common_ancestor is not None:
                print(
                    f"Lowest common ancestor of {path_a.processes} and {path_b.processes} is {common_ancestor.name}")

                # Locate the probability history entry corresponding to the last probability BEFORE the common ancestor
                prob_scaling_a = find_prob_before_lca(prob_history_a, parent_a, common_ancestor)
                prob_scaling_b = find_prob_before_lca(prob_history_b, parent_b, common_ancestor)
                prob_scaling = min(prob_scaling_a, prob_scaling_b)

                print(f"Probability Scaling Factor: {prob_scaling}")

                if common_ancestor.node_type == "Bernoulli":
                    prob = get_lca_probability(common_ancestor, parent_a, parent_b)
                    k_ab = -prob * (1 - prob)
                    print(f"Cross covariance is: {k_ab}")
                    W_ab = - path_gains[i] * path_gains[j] * path_transfers[i] * path_transfers[j] * k_ab \
                           * (signal.wiener * prob_scaling - signal.mean_quanta * prob_scaling)

                elif common_ancestor.node_type == "Fork":
                    print("Cross covariance is 0 (Fork Branch)")
                    W_ab = path_gains[i] * path_gains[j] * path_transfers[i] * path_transfers[
                        j] * signal.wiener * prob_scaling

                combined_signal.wiener += 2 * W_ab

    return combined_signal


def find_lowest_common_ancestor(node_a, node_b):
    """
    Finds the lowest common ancestor (LCA) between two Nodes in the parallel branching tree.
    """
    ancestors_a = set()
    while node_a is not None:
        ancestors_a.add(node_a)
        node_a = node_a.parent

    while node_b is not None:
        if node_b in ancestors_a:
            return node_b
        node_b = node_b.parent
    return None


def get_lca_probability(lca_node, parent_a, parent_b):
    """
    Retrieves the probability at the Lowest Common Ancestor (LCA). The output is used to calculate the cross-covariance
    of two Paths related by a Bernoulli branch.

    :param lca_node: The lowest common ancestor Node.
    :param parent_a: Parent Node of the first Path.
    :param parent_b: Parent Node of the second Path.
    :return: Probability at the LCA.
    """
    if lca_node.node_type == "Fork":
        return 1.0

    elif lca_node.node_type == "Bernoulli":
        for branch, prob in lca_node.branches:
            if branch == parent_a or branch == parent_b:
                return prob

    return 1.0


def find_prob_before_lca(prob_history, parent, common_ancestor):
    node = parent
    depth = len(prob_history) - 1
    # Iterate until we reach the initial node (which has no parent)
    while node is not None:
        if node == common_ancestor:
            return prob_history[depth - 1] if depth > 0 else 1.0
        # If the current Node is not a common ancestor, go one level up.
        node = node.parent
        depth -= 1
    return 1.0
