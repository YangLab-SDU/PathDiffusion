"""SO(3) diffusion methods."""
"""
----------------
MIT License

Copyright (c) 2022 Jason Yim, Brian L Trippe, Valentin De Bortoli, Emile Mathieu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
----------------
This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”). 
All Bytedance's Modifications are Copyright (2024) Bytedance Ltd. and/or its affiliates. 

----------------
This file may have been modified by Zhao Kailong et al.
"""
import numpy as np
import os

from scipy.spatial.transform import Rotation
import logging
import torch

from src.utils import hydra_utils
logger = hydra_utils.get_pylogger(__name__)


def compose_rotvec(r1, r2):
    """Compose two rotation euler vectors."""
    R1 = rotvec_to_matrix(r1)
    R2 = rotvec_to_matrix(r2)
    cR = np.einsum("...ij,...jk->...ik", R1, R2)
    return matrix_to_rotvec(cR)


def rotvec_to_matrix(rotvec):
    return Rotation.from_rotvec(rotvec).as_matrix()


def matrix_to_rotvec(mat):
    return Rotation.from_matrix(mat).as_rotvec()


move_to_np = lambda x: x.cpu().detach().numpy()


def igso3_expansion(omega, eps, L=1000, use_torch=False):
    """Truncated sum of IGSO(3) distribution."""
    lib = torch if use_torch else np

    if not use_torch:
        if omega.ndim == 1:
            omega = omega[np.newaxis, :]
        if eps.ndim == 1:
            eps = eps[np.newaxis, :]
    else:
        if omega.dim() == 1:
            omega = omega.unsqueeze(0)
        if eps.dim() == 1:
            eps = eps.unsqueeze(0)

    ls = lib.arange(L)
    if use_torch:
        ls = ls.to(omega.device)

    ls = ls[None, :]  # [1,L]

    if use_torch:
        omega = omega.unsqueeze(-1)  # [B,N,1]
        eps = eps.unsqueeze(-1)    # [B,N,1]
    else:
        omega = omega[..., np.newaxis]  # [B,N,1]
        eps = eps[..., np.newaxis]    # [B,N,1]

    p = (
        (2 * ls + 1)
        * lib.exp(-ls * (ls + 1) * eps ** 2 / 2)
        * lib.sin(omega * (ls + 1 / 2))
        / (lib.sin(omega / 2))
    )

    if use_torch:
        return p.sum(dim=-1)  # [B,N]
    else:
        return p.sum(axis=-1)


def density(expansion, omega, marginal=True):
    """IGSO(3) density.

    Args:
        expansion: truncated approximation of the power series in the IGSO(3)
        density.
        omega: length of an Euler vector (i.e. angle of rotation)
        marginal: set true to give marginal density over the angle of rotation,
            otherwise include normalization to give density on SO(3) or a
            rotation with angle omega.
    """
    if marginal:
        # if marginal, density over [0, pi], else over SO(3)
        return expansion * (1 - np.cos(omega)) / np.pi
    else:
        # the constant factor doesn't affect any actual calculations though
        return expansion / 8 / np.pi**2


def score(exp, omega, eps, L=1000, use_torch=False):
    """score uses the quotient rule to compute the scaling factor."""
    lib = torch if use_torch else np

    if not use_torch:  # numpy数组处理
        if exp.ndim == 1:
            exp = exp[np.newaxis, :]
        if omega.ndim == 1:
            omega = omega[np.newaxis, :]
        if eps.ndim == 1:
            eps = eps[np.newaxis, :]
    else:  # torch张量处理
        if exp.dim() == 1:
            exp = exp.unsqueeze(0)
        if omega.dim() == 1:
            omega = omega.unsqueeze(0)
        if eps.dim() == 1:
            eps = eps.unsqueeze(0)

    ls = lib.arange(L)
    if use_torch:
        ls = ls.to(exp.device)

    ls = ls[None, :]  # [1,L]

    if use_torch:
        omega = omega.unsqueeze(-1)  # [B,N,1]
        eps = eps.unsqueeze(-1)      # [B,N,1]
    else:
        omega = omega[..., np.newaxis]  # [B,N,1]
        eps = eps[..., np.newaxis]      # [B,N,1]

    hi = lib.sin(omega * (ls + 1 / 2))
    dhi = (ls + 1 / 2) * lib.cos(omega * (ls + 1 / 2))
    lo = lib.sin(omega / 2)
    dlo = 0.5 * lib.cos(omega / 2)

    dSigma = (
        (2 * ls + 1)
        * lib.exp(-ls * (ls + 1) * eps ** 2 / 2)
        * (lo * dhi - hi * dlo)
        / (lo ** 2)
    )

    if use_torch:
        return dSigma.sum(dim=-1) / (exp + 1e-4)  # [B,N]
    else:
        return dSigma.sum(axis=-1) / (exp + 1e-4)



class SO3Diffuser:

    def __init__(self, so3_conf):
        self.schedule = so3_conf.schedule

        self.min_sigma = so3_conf.min_sigma
        self.max_sigma = so3_conf.max_sigma

        self.num_sigma = so3_conf.num_sigma
        self.use_cached_score = so3_conf.use_cached_score
        self._log = logging.getLogger(__name__)

        # Discretize omegas for calculating CDFs. Skip omega=0.
        self.discrete_omega = np.linspace(0, np.pi, so3_conf.num_omega + 1)[1:]

        # Precompute IGSO3 values.
        replace_period = lambda x: str(x).replace(".", "_")
        cache_dir = os.path.join(
            so3_conf.cache_dir,
            f"eps_{so3_conf.num_sigma}_omega_{so3_conf.num_omega}_min_sigma_{replace_period(so3_conf.min_sigma)}_max_sigma_{replace_period(so3_conf.max_sigma)}_schedule_{so3_conf.schedule}",
        )

        # If cache directory doesn't exist, create it
        if not os.path.isdir(cache_dir):
            os.makedirs(cache_dir)
        pdf_cache = os.path.join(cache_dir, "pdf_vals.npy")
        cdf_cache = os.path.join(cache_dir, "cdf_vals.npy")
        score_norms_cache = os.path.join(cache_dir, "score_norms.npy")

        if (
            os.path.exists(pdf_cache)
            and os.path.exists(cdf_cache)
            and os.path.exists(score_norms_cache)
        ):
            self._log.info(f"Using cached IGSO3 in {cache_dir}")
            self._pdf = np.load(pdf_cache)
            self._cdf = np.load(cdf_cache)
            self._score_norms = np.load(score_norms_cache)
        else:
            self._log.info(f"Computing IGSO3. Saving in {cache_dir}")
            # compute the expansion of the power series
            exp_vals = np.asarray(
                [
                    igso3_expansion(self.discrete_omega, sigma)
                    for sigma in self.discrete_sigma
                ]
            )
            # Compute the pdf and cdf values for the marginal distribution of the angle
            # of rotation (which is needed for sampling)
            self._pdf = np.asarray(
                [density(x, self.discrete_omega, marginal=True) for x in exp_vals]
            )
            self._cdf = np.asarray(
                [pdf.cumsum() / so3_conf.num_omega * np.pi for pdf in self._pdf]
            )

            # Compute the norms of the scores.  This are used to scale the rotation axis when
            # computing the score as a vector.
            self._score_norms = np.asarray(
                [
                    score(exp_vals[i], self.discrete_omega, x)
                    for i, x in enumerate(self.discrete_sigma)
                ]
            )

            # Cache the precomputed values
            np.save(pdf_cache, self._pdf)
            np.save(cdf_cache, self._cdf)
            np.save(score_norms_cache, self._score_norms)

        self._score_scaling = np.sqrt(
            np.abs(
                np.sum(self._score_norms**2 * self._pdf, axis=-1)
                / np.sum(self._pdf, axis=-1)
            )
        ) / np.sqrt(3)

    @property
    def discrete_sigma(self):
        return self.sigma(np.linspace(0.0, 1.0, self.num_sigma))

    def sigma_idx(self, sigma: np.ndarray):
        """Calculates the index for discretized sigma during IGSO(3) initialization."""
        return np.digitize(sigma, self.discrete_sigma) - 1

    def sigma(self, t: np.ndarray) -> np.ndarray:
        if np.any(t < 0) or np.any(t > 1):
            raise ValueError(f"Invalid t={t} (must be in [0,1])")
        if self.schedule == "logarithmic":
            return np.log(
                t * np.exp(self.max_sigma) +
                (1 - t) * np.exp(self.min_sigma)
            )
        else:
            raise ValueError(f"Unrecognized schedule {self.schedule}")

    def diffusion_coef(self, t: np.ndarray) -> np.ndarray:
        if self.schedule == "logarithmic":
            sigma_t = self.sigma(t)  # [...,]
            numerator = 2 * (np.exp(self.max_sigma) - np.exp(self.min_sigma)) * sigma_t
            g_t = np.sqrt(numerator / np.exp(sigma_t))  # [...,]
        else:
            raise ValueError(f"Unrecognized schedule {self.schedule}")
        return g_t

    def t_to_idx(self, t: np.ndarray):
        """Helper function to go from time t to corresponding sigma_idx."""
        return self.sigma_idx(self.sigma(t))

    def sample_igso3(self, t: np.ndarray, n_samples: float = 1):
        x = np.random.rand(n_samples)

        if np.isscalar(t):
            return np.interp(x, self._cdf[self.t_to_idx(t)], self.discrete_omega)
        else:
            assert len(t) == n_samples, "t must match n_samples"
            angles = np.zeros(n_samples)
            for i in range(n_samples):
                angles[i] = np.interp(
                    x[i],
                    self._cdf[self.t_to_idx(t[i])],
                    self.discrete_omega
                )
            return angles

    def sample(self, t: np.ndarray, n_samples: float = 1):
        x = np.random.randn(n_samples, 3)
        x /= np.linalg.norm(x, axis=-1, keepdims=True)
        return x * self.sample_igso3(t, n_samples=n_samples)[:, None]

    def sample_ref(self, n_samples: float = 1, seq_len=None):
        if seq_len is not None:
            return self.sample(1, n_samples=(n_samples * seq_len)).reshape(
                n_samples, seq_len, 3
            )
        else:
            return self.sample(1, n_samples=n_samples)


    def score(self, vec: np.ndarray, t: np.ndarray, eps: float = 1e-6):
        vec_tensor = torch.tensor(vec)
        t_tensor = torch.tensor(t)

        # Call torch_score - output should be same shape as input vec
        torch_score = self.torch_score(vec_tensor, t_tensor)

        # logger.info(f"torch_score ==== {torch_score.shape}")

        return torch_score.numpy()

    def torch_score(
            self,
            vec: torch.Tensor,
            t: torch.Tensor,
            eps: float = 1e-6,
    ):
        """Computes the score of IGSO(3) density as a rotation vector."""
        original_shape = vec.shape

        vec_flat = vec.reshape(-1, 3)
        t_flat = t.reshape(-1)

        omega = torch.linalg.norm(vec_flat, dim=-1, keepdim=True) + eps  # [B*L,1]

        if self.use_cached_score:
            t_np = move_to_np(t_flat)
            idx = self.t_to_idx(t_np)
            score_norms_t = torch.tensor(self._score_norms[idx]).to(vec.device)

            omega_bins = torch.tensor(self.discrete_omega[:-1]).to(vec.device)
            omega_idx = torch.bucketize(omega.squeeze(-1), omega_bins)

            omega_scores_t = torch.gather(
                score_norms_t,
                dim=2,
                index=omega_idx.unsqueeze(1).unsqueeze(1)
            ).squeeze(-1).squeeze(-1)
        else:
            t_np = move_to_np(t_flat)
            if np.isscalar(t_np):
                sigma = self.discrete_sigma[self.t_to_idx(t_np)]
                sigma = torch.tensor(sigma).to(vec.device).expand(len(t_flat))
            else:
                sigma = np.array([self.discrete_sigma[self.t_to_idx(t_val)] for t_val in t_np])
                sigma = torch.tensor(sigma).to(vec.device)

            omega_ = omega.squeeze(-1)  # [B*L]
            sigma_ = sigma  # [B*L]

            omega_vals = igso3_expansion(omega_, sigma_, use_torch=True)  # [B*L]
            omega_scores_t = score(omega_vals, omega_, sigma_, use_torch=True)  # [B*L]

        scores = (omega_scores_t.unsqueeze(-1) * vec_flat) / (omega + eps)  # [B*L,3]
        return scores.reshape(original_shape)


    def score_scaling(self, t: np.ndarray):
        """Calculates scaling used for scores during trianing."""
        return self._score_scaling[self.t_to_idx(t)]

    def forward_marginal(self, rot_0: np.ndarray, t: np.ndarray):
        n_samples = np.cumprod(rot_0.shape[:-1])[-1]
        assert len(t) == n_samples, "t must match number of rotations"

        sampled_rots = self.sample(t, n_samples=n_samples)
        rot_score = self.score(sampled_rots, t).reshape(rot_0.shape)

        rot_t = compose_rotvec(rot_0, sampled_rots).reshape(rot_0.shape)
        return rot_t, rot_score

    def reverse(
            self,
            rot_t: np.ndarray,
            score_t: np.ndarray,
            t: np.ndarray,  # [B, L]
            dt: np.ndarray,  # [B, L]
            mask: np.ndarray = None,
            noise_scale: float = 1.0,
    ):
        batch_size, seq_len = t.shape
        rot_t = rot_t.reshape(batch_size, seq_len, 3)
        score_t = score_t.reshape(batch_size, seq_len, 3)

        if not isinstance(score_t, np.ndarray):
            score_t = score_t.cpu().numpy()
        g_t = self.diffusion_coef(t)  # [B, L]
        z = noise_scale * np.random.normal(size=score_t.shape)

        perturb = (g_t[..., None] ** 2) * score_t * dt[..., None] + g_t[..., None] * np.sqrt(dt[..., None]) * z

        if mask is not None:
            perturb *= mask[..., None]

        rot_t_1 = np.zeros_like(rot_t)
        for b in range(batch_size):
            for l in range(seq_len):
                rot_t_1[b, l] = compose_rotvec(rot_t[b, l], perturb[b, l])

        return rot_t_1.reshape(batch_size, seq_len, 3)
