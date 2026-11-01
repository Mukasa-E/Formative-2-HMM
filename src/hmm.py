# usage: python -m src.hmm

import numpy as np


class DiscreteHMM:
    """
    Discrete Hidden Markov Model.

    Parameters
    ----------
    n_hidden : int
        Number of hidden states (activities).
    n_obs : int
        Number of discrete observation symbols.
    random_state : int or None
        Seed for reproducible initialisation.
    """

    def __init__(self, n_hidden=4, n_obs=20, random_state=42):
        self.n_hidden = n_hidden
        self.n_obs = n_obs
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

        self.startprob_ = None
        self.transmat_ = None
        self.emissionprob_ = None

        self.log_likelihoods_ = []
        self.converged_ = False
        self.n_iter_ = 0

    
    def _init_parameters(self):
        """Random but valid initialisation of A, B, pi."""
        rng = self.rng

    
        pi = np.ones(self.n_hidden) / self.n_hidden
        pi += 0.01 * rng.randn(self.n_hidden)
        pi = np.clip(pi, 1e-6, None)
        self.startprob_ = pi / pi.sum()

    
        A = np.eye(self.n_hidden) * 0.8
        off = (0.2 / (self.n_hidden - 1)) if self.n_hidden > 1 else 0.0
        A += off
        A += 0.01 * rng.randn(self.n_hidden, self.n_hidden)
        A = np.clip(A, 1e-6, None)
        self.transmat_ = A / A.sum(axis=1, keepdims=True)

        
        B = np.ones((self.n_hidden, self.n_obs)) / self.n_obs
        B += 0.01 * rng.randn(self.n_hidden, self.n_obs)
        B = np.clip(B, 1e-6, None)
        self.emissionprob_ = B / B.sum(axis=1, keepdims=True)

    
    @staticmethod
    def encode_observations(X, n_obs=20, random_state=42):
        """
        Discretise continuous feature vectors into observation symbols.

        Each feature is binned into equal-frequency quantiles, then the per-dimension
        digits are combined into a single symbol index via base-levels encoding.

        Parameters
        ----------
        X : (n_windows, n_features) array
        n_obs : int, number of discrete symbols
        random_state : int

        Returns
        -------
        symbols : (n_windows,) int array in [0, n_obs)
        bin_edges : list of arrays, one per feature
        """
        rng = np.random.RandomState(random_state)
        n_features = X.shape[1]

        # Bin each feature into n_obs^(1/n_features) levels per dim
        levels = int(round(n_obs ** (1.0 / n_features)))
        levels = max(levels, 2)

        bin_edges = []
        digitised = np.zeros_like(X, dtype=int)
        for j in range(n_features):
            col = X[:, j]
            edges = np.quantile(col, np.linspace(0, 1, levels + 1))
            edges = np.unique(edges)
            if len(edges) < 2:
                edges = np.array([col.min() - 1e-6, col.max() + 1e-6])
            bin_edges.append(edges)
            digitised[:, j] = np.digitize(col, edges[1:-1])  

        
        symbols = np.zeros(len(X), dtype=int)
        for j in range(n_features):
            symbols = symbols * levels + digitised[:, j]

        
        symbols = np.clip(symbols, 0, n_obs - 1)
        return symbols, bin_edges

    
    def _forward(self, X_symbols):
        """
        Scaled forward pass.

        Returns
        -------
        alpha : (T, N) array — scaled forward probabilities (each row sums to 1)
        loglik : float — log-likelihood of the sequence
        c : (T,) array — scaling factors
        """
        T = len(X_symbols)
        N = self.n_hidden

        alpha = np.zeros((T, N))
        c = np.zeros(T)

        alpha[0] = self.startprob_ * self.emissionprob_[:, X_symbols[0]]
        c[0] = alpha[0].sum()
        if c[0] > 0:
            alpha[0] /= c[0]

        for t in range(1, T):
            alpha[t] = (alpha[t - 1] @ self.transmat_) * self.emissionprob_[:, X_symbols[t]]
            c[t] = alpha[t].sum()
            if c[t] > 0:
                alpha[t] /= c[t]

        # Avoid log(0)
        c_safe = np.where(c > 0, c, 1e-10)
        loglik = np.sum(np.log(c_safe))
        return alpha, loglik, c

    def score(self, X_symbols):
        """Log-likelihood of an observation sequence."""
        _, loglik, _ = self._forward(X_symbols)
        return loglik

    
    def predict(self, X_symbols):
        """
        Viterbi decoding — most likely hidden state sequence.

        Returns
        -------
        states : (T,) int array of state indices
        """
        T = len(X_symbols)
        N = self.n_hidden

        log_start = np.log(self.startprob_ + 1e-12)
        log_trans = np.log(self.transmat_ + 1e-12)
        log_emit = np.log(self.emissionprob_ + 1e-12)

        delta = np.zeros((T, N))
        psi = np.zeros((T, N), dtype=int)

        delta[0] = log_start + log_emit[:, X_symbols[0]]

        for t in range(1, T):
            for j in range(N):
                trans_scores = delta[t - 1] + log_trans[:, j]
                psi[t, j] = np.argmax(trans_scores)
                delta[t, j] = trans_scores[psi[t, j]] + log_emit[j, X_symbols[t]]

        # Backtrack
        states = np.zeros(T, dtype=int)
        states[T - 1] = np.argmax(delta[T - 1])
        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]

        return states

    def predict_proba(self, X_symbols):
        """
        Posterior state probabilities gamma_t(i) = P(z_t = i | X).
        Uses scaled forward-backward with consistent scaling factors.
        """
        T = len(X_symbols)
        N = self.n_hidden

        alpha, _, c = self._forward(X_symbols)

        # Backward pass with the SAME scaling factors c
        beta = np.zeros((T, N))
        beta[T - 1] = 1.0 / c[T - 1] if c[T - 1] > 0 else 1.0

        for t in range(T - 2, -1, -1):
            # beta_scaled[t][i] = sum_j(A[i][j] * B[j][x[t+1]] * beta_scaled[t+1][j]) / c[t]
            temp = self.transmat_ @ (self.emissionprob_[:, X_symbols[t + 1]] * beta[t + 1])
            ct = c[t] if c[t] > 0 else 1e-10
            beta[t] = temp / ct

        gamma = alpha * beta
        #normalise for safety
        gamma_sum = gamma.sum(axis=1, keepdims=True)
        gamma_sum = np.where(gamma_sum > 0, gamma_sum, 1e-10)
        gamma /= gamma_sum
        return gamma

    
    def fit(self, X_symbols, lengths=None, n_iter=200, tol=1e-4, verbose=True):
        """
        Baum-Welch EM algorithm with log-likelihood convergence check.

        Parameters
        ----------
        X_symbols : (n_total,) int array — concatenated observation sequences
        lengths : list of int — length of each sequence (for multi-file training)
        n_iter : int, maximum EM iterations
        tol : float, stop when log-likelihood improvement < tol
        verbose : bool

        Returns
        -------
        self
        """
        if lengths is None:
            lengths = [len(X_symbols)]

        self._init_parameters()
        self.log_likelihoods_ = []
        self.converged_ = False
        self.n_iter_ = 0

        #split concatenated symbols into sequences
        sequences = []
        idx = 0
        for L in lengths:
            sequences.append(X_symbols[idx:idx + L])
            idx += L

        prev_loglik = -np.inf

        for iteration in range(n_iter):
            self.n_iter_ = iteration + 1

            # Accumulators
            start_num = np.zeros(self.n_hidden)
            trans_num = np.zeros((self.n_hidden, self.n_hidden))
            trans_den = np.zeros(self.n_hidden)
            emit_num = np.zeros((self.n_hidden, self.n_obs))
            emit_den = np.zeros(self.n_hidden)

            total_loglik = 0.0

            for seq in sequences:
                T = len(seq)
                if T < 2:
                    continue

                # Scaled forward pass
                alpha, loglik, c = self._forward(seq)
                total_loglik += loglik

                # Scaled backward pass using the same scaling factors c
                beta = np.zeros((T, self.n_hidden))
                beta[T - 1] = 1.0 / c[T - 1] if c[T - 1] > 0 else 1.0

                for t in range(T - 2, -1, -1):
                    temp = self.transmat_ @ (
                        self.emissionprob_[:, seq[t + 1]] * beta[t + 1]
                    )
                    ct = c[t] if c[t] > 0 else 1e-10
                    beta[t] = temp / ct

                # gamma[t][i] = P(z_t = i | X) = alpha[t][i] * beta[t][i]
                gamma = alpha * beta
                gamma_sum = gamma.sum(axis=1, keepdims=True)
                gamma_sum = np.where(gamma_sum > 0, gamma_sum, 1e-10)
                gamma /= gamma_sum

                # xi[t][i][j] = P(z_t=i, z_{t+1}=j | X)
                xi = np.zeros((T - 1, self.n_hidden, self.n_hidden))
                for t in range(T - 1):
                    for i in range(self.n_hidden):
                        for j in range(self.n_hidden):
                            xi[t, i, j] = (
                                alpha[t, i]
                                * self.transmat_[i, j]
                                * self.emissionprob_[j, seq[t + 1]]
                                * beta[t + 1, j]
                            )
                    xi_t_sum = xi[t].sum()
                    if xi_t_sum > 0:
                        xi[t] /= xi_t_sum

                #Accumulate sufficient statistics
                start_num += gamma[0]
                trans_num += xi.sum(axis=0)
                trans_den += gamma[:-1].sum(axis=0)
                for t in range(T):
                    emit_num[:, seq[t]] += gamma[t]
                emit_den += gamma.sum(axis=0)

            
            self.startprob_ = start_num / start_num.sum()
            self.startprob_ = np.clip(self.startprob_, 1e-6, None)
            self.startprob_ /= self.startprob_.sum()

            self.transmat_ = trans_num / trans_den[:, None]
            self.transmat_ = np.clip(self.transmat_, 1e-6, None)
            self.transmat_ /= self.transmat_.sum(axis=1, keepdims=True)

            self.emissionprob_ = emit_num / emit_den[:, None]
            self.emissionprob_ = np.clip(self.emissionprob_, 1e-6, None)
            self.emissionprob_ /= self.emissionprob_.sum(axis=1, keepdims=True)

            self.log_likelihoods_.append(total_loglik)

            if verbose and (iteration % 10 == 0 or iteration == n_iter - 1):
                print(f"  Iter {iteration + 1:3d}: log-likelihood = {total_loglik:.4f}")

            # Convergence check: log-likelihood improvement < epsilon
            if abs(total_loglik - prev_loglik) < tol:
                self.converged_ = True
                if verbose:
                    print(f"  Converged at iteration {iteration + 1} "
                          f"(delta log-likelihood = {abs(total_loglik - prev_loglik):.2e} < {tol})")
                break

            prev_loglik = total_loglik

        return self

    @staticmethod
    def _logsumexp(x):
        """Numerically stable log(sum(exp(x)))."""
        m = np.max(x)
        if m == -np.inf:
            return -np.inf
        return m + np.log(np.sum(np.exp(x - m)))


if __name__ == "__main__":
    rng = np.random.RandomState(0)
    n_obs = 20
    seq_lengths = [50, 60, 55]
    X = np.concatenate([rng.randint(0, n_obs, size=L) for L in seq_lengths])

    model = DiscreteHMM(n_hidden=4, n_obs=n_obs, random_state=42)
    model.fit(X, lengths=seq_lengths, n_iter=100, tol=1e-4, verbose=True)
    print(f"Converged: {model.converged_}, iterations: {model.n_iter_}")
    states = model.predict(X[:50])
    print(f"Predicted states (first 50): {states}")