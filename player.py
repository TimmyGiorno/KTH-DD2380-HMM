#!/usr/bin/env python3

"""
Code for KTH DD2380 HT24 Assignment: Fishing Derby HHM.

Members：Bingchu Zhao (Timmy), Yiyao Zhang
Email：bingchu@kth.se, yiyaoz@kth.se
"""

from player_controller_hmm import PlayerControllerHMMAbstract
import constants as CONSTANTS
import random
import numpy as np
import sys
import typing
from typing import List, Tuple

epsilon = 1e-12

def generate_probability_vector(
    size: int,
) -> typing.List[float]:
    """
    Generate a random probability vector.
    The sum of all entries in the probability vector is 1.
    """
    M = [(1 / size) + (np.random.rand() / 1000) for _ in range(size)]
    return [m / sum(M) for m in M]

def dot_product(matrix_a: List[List[float]], matrix_b: List[float]) -> List[List[float]]:
    """
    Compute the dot product of a row vector (matrix_a) and a column vector (matrix_b).
    """
    return [[a * b for a, b in zip(matrix_a[0], matrix_b)]]

def alpha_forward(
    A: List[List[float]],
    B: List[List[float]],
    pi: List[List[float]],
    seq: List[int],
    N: int,
    T: int,
) -> Tuple[List[List[float]], List[float]]:
    """
    Compute the forward probabilities (alpha) and scaling factors for an HMM.

    Args:
        A: Transition probability matrix.
        B: Emission probability matrix.
        pi: Initial probability distribution.
        seq: Observation sequence.
        N: Number of states.
        T: Length of the observation sequence.

    Returns:
        A tuple containing the list of alpha values and scaling factors.
    """
    alpha_list = []  # List to store alpha values at each time step
    scaling_factors = []  # List to store scaling factors

    for t in range(T):
        scaling_factor = 0
        alpha_values = []
        initial_prob = pi[0]

        for i in range(N):
            if t == 0:
                alpha = initial_prob[i] * B[i][seq[t]]
                scaling_factor += alpha
                alpha_values.append(alpha)
            else:
                alpha = 0
                for j in range(N):
                    alpha += (
                        alpha_list[t - 1][j]
                        * A[j][i]
                        * B[i][seq[t]]
                    )
                scaling_factor += alpha
                alpha_values.append(alpha)

        # Apply scaling to avoid numerical underflow
        scale = 1 / (scaling_factor + epsilon)
        for i in range(N):
            alpha_values[i] *= scale

        scaling_factors.append(scale)
        alpha_list.append(alpha_values)

    return alpha_list, scaling_factors

def beta_forward(
    a: List[List[float]],
    b: List[List[float]],
    p: List[List[float]],
    seq: List[int],
    c: List[float],
    N: int,
    T: int,
) -> List[List[float]]:
    """
    Compute the backward probabilities (beta) for an HMM.
    """
    beta_list = []  # List to store beta values

    for t in range(T):
        beta_temp_list = []
        for i in range(N):
            if t == 0:
                beta = c[t]
                beta_temp_list.append(beta)
            else:
                sum_term = 0
                for j in range(N):
                    sum_term += beta_list[t - 1][j] * a[i][j] * b[j][seq[t - 1]]
                beta_temp_list.append(sum_term)

        if t > 0:
            for m in range(N):
                beta_temp_list[m] = c[t] * beta_temp_list[m]

        beta_list.append(beta_temp_list)

    return beta_list

def compute_gamma(
    a: List[List[float]],
    b: List[List[float]],
    seq: List[int],
    alpha_list: List[List[float]],
    beta_list: List[List[float]],
    N: int,
    T: int,
) -> Tuple[List[List[float]], List[List[List[float]]]]:
    """
    Compute the gamma and di-gamma values for re-estimating HMM parameters.
    """
    gama_list = []
    gama_ij_list = []

    for t in range(T - 1):
        gama_temp_list = []
        gama_ij_temp_list = []

        for i in range(N):
            gama_val_temp = []
            gama = 0
            for j in range(N):
                gama_ij = (
                    alpha_list[t][i]
                    * a[i][j]
                    * b[j][seq[t + 1]]
                    * beta_list[t + 1][j]
                )
                gama += gama_ij
                gama_val_temp.append(gama_ij)
            gama_temp_list.append(gama)
            gama_ij_temp_list.append(gama_val_temp)

        gama_list.append(gama_temp_list)
        gama_ij_list.append(gama_ij_temp_list)

    # Handle final gamma for last time step
    gama_temp_list = []
    alpha_temp_list = alpha_list[T - 1]
    for k in range(N):
        gama_temp_list.append(alpha_temp_list[k])
    gama_list.append(gama_temp_list)

    return gama_list, gama_ij_list


def reestimate_model(
    gama_list: List[List[float]],
    gama_ij_list: List[List[List[float]]],
    seq: List[int],
    M: int,
    N: int,
    T: int,
) -> Tuple[List[List[float]], List[List[float]], List[List[float]]]:
    """
    Re-estimate the HMM parameters (PI, A, B) using the gamma and di-gamma values.
    """
    # Re-estimate initial state distribution
    pi_temp_list = [gama_list[0][i] for i in range(N)]

    # Re-estimate transition matrix A
    trans_mat_new = []
    for i in range(N):
        denom = sum(gama_list[t][i] for t in range(T - 1))
        trans_mat_temp_list = [
            sum(gama_ij_list[t][i][j] for t in range(T - 1)) / (denom + epsilon)
            for j in range(N)
        ]
        trans_mat_new.append(trans_mat_temp_list)

    # Re-estimate observation matrix B
    obs_mat_new = []
    for i in range(N):
        denom = sum(gama_list[t][i] for t in range(T))
        obs_mat_temp_list = [
            sum(gama_list[t][i] for t in range(T) if seq[t] == j) / (denom + epsilon)
            for j in range(M)
        ]
        obs_mat_new.append(obs_mat_temp_list)

    return [pi_temp_list], trans_mat_new, obs_mat_new

def log_probability(
    c: List[float],
    T: int,
) -> float:
    """
    Compute the logarithm of the probability of the observation sequence.
    """
    return -sum(np.log(c[t]) for t in range(T))

def transpose(
    M: List[List[float]],
) -> List[List[float]]:
    """
    Compute the transpose of a matrix.
    """
    return [list(i) for i in zip(*M)]

def matrix_multiplication(
    A: List[List[float]],
    B: List[List[float]],
) -> List[List[float]]:
    """
    Compute the matrix multiplication of A and B.
    """
    return [[sum(a * b for a, b in zip(a_row, b_col)) for b_col in zip(*B)] for a_row in A]


class HMM:
    """
    Hidden Markov Model for Fishing Derby HMM. You cannot observe the type of fish directly.
    Instead, you observe the sequence of actions (emissions) of it and guess it.

    - State: Type of fish.
    - Emissions: Movements of fish.
    """
    def __init__(
        self,
        num_states: int,
        num_emissions: int,
    ) -> None:
        self.PI = [generate_probability_vector(num_states)]
        self.A = [generate_probability_vector(num_states) for _ in range(num_states)]
        self.B = [generate_probability_vector(num_emissions) for _ in range(num_states)]
        return

    def forward_algorithm(
        self,
        fish_observation: List[int],
    ) -> float:
        transposed_B = transpose(self.B)
        alpha = dot_product(self.PI, transposed_B[fish_observation[0]])
        for e in fish_observation[1:]:
            alpha = matrix_multiplication(alpha, self.A)
            alpha = dot_product(alpha, transposed_B[e])
        return sum(alpha[0])

    def baum_welch(
        self,
        observation_seq,
    ) -> None:

        N = len(self.A)
        M = len(observation_seq)
        T = len(observation_seq)

        # initial parameters.
        num_iteration = 0
        max_iterations = 5
        old_log_pr = float("-inf")
        log_pr = 1

        # Before the maximum number of iterations is reached
        # and when the logarithmic probability of the observed sequence is still increasing
        while num_iteration < max_iterations and log_pr > old_log_pr:
            num_iteration += 1
            if num_iteration != 1:
                old_log_pr = log_pr

            # alpha-forward.
            alpha_vals, c_val = alpha_forward(self.A, self.B, self.PI, observation_seq, N, T)

            # beta-backward.
            c_beta = c_val[::-1]
            seq_beta = observation_seq[::-1]
            beta_flip = beta_forward(self.A, self.B, self.PI, seq_beta, c_beta, N, T)
            beta_vals = beta_flip[::-1]

            # Di-gamma.
            gamma_list, gamma_ij_list = compute_gamma(
                self.A, self.B, observation_seq, alpha_vals, beta_vals, N, T)

            # Re-estimate lambda.
            PI, A, B = reestimate_model(gamma_list, gamma_ij_list, observation_seq, M, N, T)

        self.PI, self.A, self.B = PI, A, B
        return


class PlayerControllerHMM(PlayerControllerHMMAbstract):
    def init_parameters(
        self,
    ) -> None:
        self.fishes_hmms = [
            HMM(num_states=1, num_emissions=CONSTANTS.N_EMISSIONS) for _ in range(CONSTANTS.N_SPECIES)
        ]
        self.fishes_observations = [(i, []) for i in range(CONSTANTS.N_FISH)]
        self.fish_observation = None

    def guess(
        self,
        step_count: int,
        observations: List[int],
    ) -> Tuple[int, int]:

        for i in range(len(self.fishes_observations)):
            self.fishes_observations[i][1].append(observations[i])

        # Each game contains 70 fish and has a maximal duration of 180 time steps.
        # The game will end when the player has made a guess 70 times.
        # Instead, let fish move before t=110 and then collect their actions.
        if step_count < 110:
            return None
        else:
            fish_id, fish_observation = self.fishes_observations.pop()
            fish_type = 0
            max_pr = 0
            for model, j in zip(self.fishes_hmms, range(CONSTANTS.N_SPECIES)):
                forward_pr = model.forward_algorithm(fish_observation)
                if forward_pr > max_pr:
                    max_pr = forward_pr
                    fish_type = j
            self.fish_observation = fish_observation
            return fish_id, fish_type

    def reveal(
        self,
        correct: bool,
        fish_id: int,
        true_type: int,
    ) -> None:
        if not correct:
            self.fishes_hmms[true_type].baum_welch(self.fish_observation)
        else:
            return
