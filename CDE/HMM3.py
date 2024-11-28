#!/usr/bin/env python3

"""
Code for KTH DD2380 HT24 Assignment: HMM3.

Members: Bingchu Zhao (Timmy), Yiyao Zhang
Email: bingchu@kth.se, yiyaoz@kth.se
"""
import math
import time

class HMMModel:
    def __init__(self, A, B, pi):
        self.A = A  # Transition probabilities
        self.B = B  # Emission probabilities
        self.pi = pi  # Initial state distribution

    @staticmethod
    def matrix_multiply(A, B):
        """
        Matrix multiplication of two 2D lists A and B.
        """
        return [[sum(a * b for a, b in zip(A_row, B_col)) for B_col in zip(*B)] for A_row in A]

    def emissions_probability(self):
        """
        Calculate emissions probability as multiply(pi, A, B).
        """
        return self.matrix_multiply(self.matrix_multiply(self.pi, self.A), self.B)

    def forward_algorithm(self, obs):
        """
        Calculate the forward probability using the given observations.
        """
        alpha = [[self.pi[0][i] * self.B[i][obs[0]] for i in range(len(self.pi[0]))]]
        for t in range(1, len(obs)):
            alpha.append([sum(alpha[t-1][j] * self.A[j][i] for j in range(len(self.A))) * self.B[i][obs[t]] for i in range(len(self.A))])
        return sum(alpha[-1])

    def viterbi(self, obs):
        """
        Perform the Viterbi algorithm to determine the most probable state sequence.
        """
        n, T = len(self.A), len(obs)
        delta = [[0] * n for _ in range(T)]
        delta_idx = [[0] * n for _ in range(T)]

        # Initialization
        for i in range(n):
            delta[0][i] = self.pi[0][i] * self.B[i][obs[0]]

        # Recursion
        for t in range(1, T):
            for j in range(n):
                delta[t][j] = max(delta[t-1][i] * self.A[i][j] for i in range(n)) * self.B[j][obs[t]]
                delta_idx[t][j] = max(range(n), key=lambda i: delta[t-1][i] * self.A[i][j])

        # Backtracking
        X = [0] * T
        X[-1] = max(range(n), key=lambda i: delta[T-1][i])
        for t in range(T-2, -1, -1):
            X[t] = delta_idx[t+1][X[t+1]]

        return X

    def baum_welch(self, obs, max_time=0.8):
        """
        Perform the Baum-Welch algorithm to optimize A, B, and pi.
        """
        def re_estimate(A, B, pi, obs):
            alpha, scalers = self.alpha_pass(A, B, pi, obs)
            beta = self.beta_pass(A, B, obs, scalers)
            gamma, di_gamma = self.get_gammas(A, B, alpha, beta, obs)

            new_pi = [[gamma[0][i] for i in range(len(A))]]
            new_A = [[sum(di_gamma[t][i][j] for t in range(len(obs)-1)) / sum(gamma[t][i] for t in range(len(obs)-1)) for j in range(len(A))] for i in range(len(A))]
            new_B = [[sum(gamma[t][j] for t in range(len(obs)) if obs[t] == k) / sum(gamma[t][j] for t in range(len(obs))) for k in range(len(B[0]))] for j in range(len(A))]

            return new_A, new_B, new_pi, scalers

        start_time = time.time()
        A, B, pi = self.A, self.B, self.pi
        previous_log_likelihood = float("-inf")

        while True:
            new_A, new_B, new_pi, scalers = re_estimate(A, B, pi, obs)
            current_log_likelihood = -sum(math.log(s) for s in scalers)

            if current_log_likelihood <= previous_log_likelihood or time.time() - start_time > max_time:
                break

            A, B, pi = new_A, new_B, new_pi
            previous_log_likelihood = current_log_likelihood

        return new_A, new_B, new_pi

    def alpha_pass(self, A, B, pi, obs):
        """
        Perform the alpha-pass (forward pass) with scaling to prevent underflow.
        """
        alpha = [[pi[0][i] * B[i][obs[0]] for i in range(len(pi[0]))]]
        scalers = [1 / sum(alpha[0])]
        alpha[0] = [a * scalers[0] for a in alpha[0]]

        for t in range(1, len(obs)):
            alpha.append([sum(alpha[t-1][j] * A[j][i] for j in range(len(A))) * B[i][obs[t]] for i in range(len(A))])
            scalers.append(1 / sum(alpha[t]))
            alpha[t] = [a * scalers[t] for a in alpha[t]]

        return alpha, scalers

    def beta_pass(self, A, B, obs, scalers):
        """
        Perform the beta-pass (backward pass) with scaling.
        """
        beta = [[scalers[-1]] * len(A)]
        for t in range(len(obs) - 2, -1, -1):
            beta.insert(0, [sum(beta[0][j] * A[i][j] * B[j][obs[t+1]] for j in range(len(A))) for i in range(len(A))])
            beta[0] = [b * scalers[t] for b in beta[0]]
        return beta

    def get_gammas(self, A, B, alpha, beta, obs):
        """
        Compute gamma and di-gamma values.
        """
        gamma, di_gamma = [], []
        for t in range(len(obs) - 1):
            di_gamma_t = [[alpha[t][i] * A[i][j] * B[j][obs[t+1]] * beta[t+1][j] for j in range(len(A))] for i in range(len(A))]
            di_gamma.append(di_gamma_t)
            gamma.append([sum(di_gamma_t[i]) for i in range(len(A))])
        gamma.append(alpha[-1])
        return gamma, di_gamma


def parse_input():
    """
    Parse input from stdin.
    """
    lines = []
    while True:
        try:
            lines.append(input())
        except EOFError:
            break

    def parse_matrix(line):
        tokens = list(map(float, line.split()))
        n, m = int(tokens.pop(0)), int(tokens.pop(0))
        return [tokens[i*m:(i+1)*m] for i in range(n)]

    A = parse_matrix(lines[0])
    B = parse_matrix(lines[1])
    pi = parse_matrix(lines[2])
    obs = list(map(int, lines[3].split()[1:]))

    return A, B, pi, obs


def main():
    """
    Main execution point for the HMM model training and result generation.
    """
    A, B, pi, obs = parse_input()
    hmm = HMMModel(A, B, pi)

    new_A, new_B, new_pi = hmm.baum_welch(obs)

    def format_output(matrix):
        rows, cols = len(matrix), len(matrix[0])
        return f"{rows} {cols} " + " ".join(" ".join(map(str, row)) for row in matrix)

    print(format_output(new_A))
    print(format_output(new_B))


if __name__ == "__main__":
    main()
