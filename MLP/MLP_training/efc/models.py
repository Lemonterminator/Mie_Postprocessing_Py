from __future__ import annotations

"""MLP architecture and physical sigmoid penetration model.

PenetrationMLP  — heteroscedastic MLP with output_dim=2: channel 0 = mu_hat
                  (dimensionless, mean / A_scale), channel 1 = log_var_hat.
                  Architecture: Linear → LayerNorm → Activation (→ Dropout) ×
                  len(hidden_dims), then a final linear projection.
build_model     — construct from a config dict.

The physical sigmoid blends two power-law segments:
    S(t) = (1 − w) · k_sqrt · √t  +  w · k_quarter · t^0.25
where w is a soft sigmoid gate at transition time t0.  In practice k_sqrt is
negligible (ratio ≈ 3e-9 relative to k_quarter at t0 in fitted distributions);
the dominant physics is k_quarter, consistent with A_scale ∝ ΔP^0.5.
These functions are used only for preprocessing reference trajectories and
are not part of the MLP forward pass.
"""

import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .feature_registry import MIN_TIME_SHIFT_S


RESIDUAL_FILM_ARCHITECTURE_MODES = {
    "residual_film_family_head",
    "residual_film_last_block",
    "residual_film_all_blocks",
}


def make_activation(name: str) -> nn.Module:
    token = (name or "relu").lower()
    if token == "relu":
        return nn.ReLU()
    if token == "gelu":
        return nn.GELU()
    if token == "tanh":
        return nn.Tanh()
    if token == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported activation '{name}'")


class PenetrationMLP(nn.Module):
    """Fully-connected heteroscedastic MLP for spray-penetration prediction.

    Outputs two channels per time point in dimensionless A-scaled space:
      channel 0: mu_hat     — predicted mean (penetration / A_scale)
      channel 1: log_var_hat — predicted log-variance

    Physical mu and std are recovered by multiplying by A_scale in the loss
    functions.  The heteroscedastic variance head enables calibrated prediction
    intervals without post-hoc scaling.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int,
        *,
        activation: str = "gelu",
        dropout: float = 0.0,
        onset_aux_head: bool = False,
        onset_aux_hidden: int = 64,
        learnable_d2: bool = False,
        n_families_for_d2: int = 6,
        learnable_d2_floor: float = 1e-5,
        learnable_d2_init: float = 5e-4,
    ) -> None:
        super().__init__()
        self.onset_aux_head = bool(onset_aux_head)
        self.learnable_d2 = bool(learnable_d2)
        self.learnable_d2_floor = float(learnable_d2_floor)
        self.n_families_for_d2 = int(n_families_for_d2)
        if self.learnable_d2:
            target = max(float(learnable_d2_init) - float(learnable_d2_floor), 1e-9)
            inv_softplus_init = math.log(math.expm1(target))
            self.log_lambda_d2 = nn.Parameter(
                torch.full((self.n_families_for_d2,), inv_softplus_init, dtype=torch.float32)
            )
        else:
            self.log_lambda_d2 = None
        layers: list[nn.Module] = []
        in_dim = int(input_dim)
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(in_dim, int(hidden_dim)),
                    nn.LayerNorm(int(hidden_dim)),
                    make_activation(activation),
                ]
            )
            if float(dropout) > 0:
                layers.append(nn.Dropout(float(dropout)))
            in_dim = int(hidden_dim)
        if not self.onset_aux_head:
            # Backward-compatible structure: self.net carries the full stack
            # including the (mu, log_var) head. Existing checkpoints load.
            layers.append(nn.Linear(in_dim, int(output_dim)))
            self.net = nn.Sequential(*layers)
        else:
            # Trunk + (mu, log_var) head + parallel onset head. Trunk is the
            # same width sequence; the final Linear for (mu, log_var) is split
            # off so the onset head can branch from the same embedding.
            self.trunk = nn.Sequential(*layers)
            self.main_head = nn.Linear(in_dim, int(output_dim))
            self.onset_head = nn.Sequential(
                nn.Linear(in_dim, int(onset_aux_hidden)),
                make_activation(activation),
                nn.Linear(int(onset_aux_hidden), 1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.onset_aux_head:
            return self.net(x)
        h = self.trunk(x)
        main = self.main_head(h)
        onset = self.onset_head(h)
        return torch.cat([main, onset], dim=-1)


class FamilyAwarePenetrationMLP(nn.Module):
    """Shared trunk + per-family mu heads (Tier-3A architecture).

    Family routing assumes the LAST feature channel is an integer family_id in
    [0, n_families). It is stripped before the trunk, used to dispatch to the
    correct mu head, and never z-scored. The log_var head is shared across
    families on purpose: factory-fresh (family 0) has only one nozzle of data,
    not enough to learn variance reliably.

    Fallback for LONO-N0 (factory-fresh held out): the `trained_families`
    buffer marks which heads have actually seen training data. Untrained
    families at inference time route to `fallback_family_id` (default: 1,
    which always has training data in this 2-family scheme).

    Output: [..., 2] (mu, log_var) so the model is drop-in compatible with
    PenetrationMLP downstream consumers (split_mu_logvar, etc.).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int = 2,
        *,
        activation: str = "gelu",
        dropout: float = 0.0,
        n_families: int = 2,
        family_head_dims: Sequence[int] = (128,),
        fallback_family_id: int = 1,
    ) -> None:
        super().__init__()
        self.n_families = int(n_families)
        if int(output_dim) != 2:
            raise ValueError(
                "FamilyAwarePenetrationMLP currently produces 2 channels (mu, log_var) only."
            )
        self.fallback_family_id = int(fallback_family_id)

        trunk_in_dim = int(input_dim) - 1  # last channel is family_id
        if trunk_in_dim <= 0:
            raise ValueError(
                f"FamilyAwarePenetrationMLP needs input_dim >= 2 (got {input_dim})."
            )

        # Shared trunk (Linear -> LayerNorm -> Activation [-> Dropout]) x len(hidden_dims)
        trunk_layers: list[nn.Module] = []
        prev = trunk_in_dim
        for h in hidden_dims:
            trunk_layers.extend(
                [
                    nn.Linear(prev, int(h)),
                    nn.LayerNorm(int(h)),
                    make_activation(activation),
                ]
            )
            if float(dropout) > 0:
                trunk_layers.append(nn.Dropout(float(dropout)))
            prev = int(h)
        self.trunk = nn.Sequential(*trunk_layers)
        trunk_out = prev

        # Per-family mu heads: a small MLP each so the family head has some
        # capacity beyond a single Linear, per plan section 3.1.
        def _build_head(out_dim: int) -> nn.Module:
            head_layers: list[nn.Module] = []
            inner_prev = trunk_out
            for h in family_head_dims:
                head_layers.extend([nn.Linear(inner_prev, int(h)), make_activation(activation)])
                inner_prev = int(h)
            head_layers.append(nn.Linear(inner_prev, int(out_dim)))
            return nn.Sequential(*head_layers)

        self.mu_heads = nn.ModuleList([_build_head(1) for _ in range(self.n_families)])
        self.log_var_head = _build_head(1)
        self.register_buffer(
            "trained_families",
            torch.zeros(self.n_families, dtype=torch.bool),
        )

    def mark_family_trained(self, family_id: int) -> None:
        idx = int(family_id)
        if 0 <= idx < self.n_families:
            self.trained_families[idx] = True

    def _route_family_ids(self, family_id: torch.Tensor) -> torch.Tensor:
        """Replace untrained family ids with the fallback (eval mode only)."""
        if self.training:
            return family_id
        if bool(self.trained_families.all().item()):
            return family_id
        out = family_id.clone()
        for f in range(self.n_families):
            if not bool(self.trained_families[f].item()):
                out = torch.where(family_id == f, torch.full_like(family_id, self.fallback_family_id), out)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        family_id_raw = x[..., -1].long()
        features = x[..., :-1]
        h = self.trunk(features)
        log_var = self.log_var_head(h)

        if self.training:
            # Record which families actually receive gradient this step.
            unique = torch.unique(family_id_raw).tolist()
            for f in unique:
                self.mark_family_trained(int(f))

        family_id = self._route_family_ids(family_id_raw)
        mu = torch.zeros(*h.shape[:-1], 1, device=h.device, dtype=h.dtype)
        for f in range(self.n_families):
            mask = family_id == f
            if mask.any():
                mu[mask] = self.mu_heads[f](h[mask])
        return torch.cat([mu, log_var], dim=-1)


class ResidualFamilyAwarePenetrationMLP(nn.Module):
    """Shared mu head plus per-family residual deltas.

    The last input channel is an integer ``family_id`` and is not passed through
    the trunk. The shared head predicts the base mean, while the selected family
    delta head predicts a correction:

        mu = mu_shared(h) + delta_family(h)

    Delta heads are zero-initialised at the final projection, so an unseen or
    newly-added family naturally falls back to the shared predictor.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int = 2,
        *,
        activation: str = "gelu",
        dropout: float = 0.0,
        n_families: int = 2,
        family_head_dims: Sequence[int] = (128,),
        fallback_family_id: int = 1,
    ) -> None:
        super().__init__()
        self.n_families = int(n_families)
        if int(output_dim) != 2:
            raise ValueError(
                "ResidualFamilyAwarePenetrationMLP currently produces 2 channels (mu, log_var) only."
            )
        self.fallback_family_id = int(fallback_family_id)

        trunk_in_dim = int(input_dim) - 1
        if trunk_in_dim <= 0:
            raise ValueError(
                f"ResidualFamilyAwarePenetrationMLP needs input_dim >= 2 (got {input_dim})."
            )

        trunk_layers: list[nn.Module] = []
        prev = trunk_in_dim
        for h in hidden_dims:
            trunk_layers.extend(
                [
                    nn.Linear(prev, int(h)),
                    nn.LayerNorm(int(h)),
                    make_activation(activation),
                ]
            )
            if float(dropout) > 0:
                trunk_layers.append(nn.Dropout(float(dropout)))
            prev = int(h)
        self.trunk = nn.Sequential(*trunk_layers)
        trunk_out = prev

        def _build_head(out_dim: int, *, zero_final: bool = False) -> nn.Module:
            head_layers: list[nn.Module] = []
            inner_prev = trunk_out
            for h in family_head_dims:
                head_layers.extend([nn.Linear(inner_prev, int(h)), make_activation(activation)])
                inner_prev = int(h)
            final = nn.Linear(inner_prev, int(out_dim))
            if zero_final:
                nn.init.zeros_(final.weight)
                nn.init.zeros_(final.bias)
            head_layers.append(final)
            return nn.Sequential(*head_layers)

        self.mu_shared_head = _build_head(1)
        self.delta_heads = nn.ModuleList(
            [_build_head(1, zero_final=True) for _ in range(self.n_families)]
        )
        self.log_var_head = _build_head(1)
        self.register_buffer(
            "trained_families",
            torch.zeros(self.n_families, dtype=torch.bool),
        )

    def mark_family_trained(self, family_id: int) -> None:
        idx = int(family_id)
        if 0 <= idx < self.n_families:
            self.trained_families[idx] = True

    def _validate_family_ids(self, family_id: torch.Tensor) -> None:
        if family_id.numel() == 0:
            return
        min_id = int(family_id.min().detach().cpu())
        max_id = int(family_id.max().detach().cpu())
        if min_id < 0 or max_id >= self.n_families:
            raise ValueError(
                f"family_id must be in [0, {self.n_families}); got min={min_id}, max={max_id}."
            )

    def forward_parts(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        family_id = x[..., -1].long()
        self._validate_family_ids(family_id)
        features = x[..., :-1]
        h = self.trunk(features)
        mu_shared = self.mu_shared_head(h)
        delta = torch.zeros(*h.shape[:-1], 1, device=h.device, dtype=h.dtype)
        for f in range(self.n_families):
            mask = family_id == f
            if mask.any():
                delta[mask] = self.delta_heads[f](h[mask])
        log_var = self.log_var_head(h)

        if self.training:
            unique = torch.unique(family_id).tolist()
            for f in unique:
                self.mark_family_trained(int(f))
        return mu_shared, delta, log_var

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu_shared, delta, log_var = self.forward_parts(x)
        return torch.cat([mu_shared + delta, log_var], dim=-1)

    def residual_delta_stats(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        _, delta, _ = self.forward_parts(x)
        delta_l2 = delta.pow(2).mean()
        return {
            "delta_l2": delta_l2,
            "delta_rms": torch.sqrt(delta_l2 + 1e-12),
        }


class FamilyFiLMAdapter(nn.Module):
    """Per-family affine modulation with identity initialization.

    This is the MLP-only ablation hook: each family can rescale and shift a
    hidden representation, while a newly-added family starts as an exact
    identity transform (gamma=1, beta=0).
    """

    def __init__(self, n_families: int, hidden_dim: int) -> None:
        super().__init__()
        self.n_families = int(n_families)
        self.hidden_dim = int(hidden_dim)
        self.gamma = nn.Parameter(torch.ones(self.n_families, self.hidden_dim))
        self.beta = nn.Parameter(torch.zeros(self.n_families, self.hidden_dim))

    def forward(self, h: torch.Tensor, family_id: torch.Tensor) -> torch.Tensor:
        flat_family = family_id.reshape(-1).long()
        gamma = self.gamma.index_select(0, flat_family).reshape(*family_id.shape, self.hidden_dim)
        beta = self.beta.index_select(0, flat_family).reshape(*family_id.shape, self.hidden_dim)
        gamma = gamma.to(device=h.device, dtype=h.dtype)
        beta = beta.to(device=h.device, dtype=h.dtype)
        return h * gamma + beta

    def l2_from_identity(self) -> torch.Tensor:
        return (self.gamma - 1.0).pow(2).mean() + self.beta.pow(2).mean()


class ResidualFiLMFamilyAwarePenetrationMLP(nn.Module):
    """Residual family head with identity-initialized hidden FiLM adapters.

    The last input channel is the integer family id. The base prediction is the
    same residual-family-head decomposition:

        mu = mu_shared(h_film) + delta_family(h_film)

    but ``h_film`` can be family-modulated by FiLM adapters after the final
    trunk block (``residual_film_last_block``) or after every trunk block
    (``residual_film_all_blocks``).  With gamma=1, beta=0, and zero residual
    heads, the model falls back to the shared predictor for a new family.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int = 2,
        *,
        activation: str = "gelu",
        dropout: float = 0.0,
        n_families: int = 2,
        family_head_dims: Sequence[int] = (128,),
        fallback_family_id: int = 1,
        film_adapter_placement: str = "last_block",
    ) -> None:
        super().__init__()
        self.n_families = int(n_families)
        if int(output_dim) != 2:
            raise ValueError(
                "ResidualFiLMFamilyAwarePenetrationMLP currently produces 2 channels (mu, log_var) only."
            )
        self.fallback_family_id = int(fallback_family_id)
        self.film_adapter_placement = _normalize_film_adapter_placement(film_adapter_placement)

        trunk_in_dim = int(input_dim) - 1
        if trunk_in_dim <= 0:
            raise ValueError(
                f"ResidualFiLMFamilyAwarePenetrationMLP needs input_dim >= 2 (got {input_dim})."
            )
        if not hidden_dims:
            raise ValueError("ResidualFiLMFamilyAwarePenetrationMLP needs at least one hidden layer.")

        trunk_layers: list[nn.Module] = []
        block_end_indices: list[int] = []
        block_dims: list[int] = []
        prev = trunk_in_dim
        for h in hidden_dims:
            hidden_dim = int(h)
            trunk_layers.extend(
                [
                    nn.Linear(prev, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    make_activation(activation),
                ]
            )
            if float(dropout) > 0:
                trunk_layers.append(nn.Dropout(float(dropout)))
            block_end_indices.append(len(trunk_layers) - 1)
            block_dims.append(hidden_dim)
            prev = hidden_dim
        self.trunk = nn.Sequential(*trunk_layers)
        trunk_out = prev

        if self.film_adapter_placement == "all_blocks":
            adapted_blocks = list(range(len(block_dims)))
        else:
            adapted_blocks = [len(block_dims) - 1]
        self._film_module_index_to_adapter = {
            block_end_indices[block_idx]: adapter_idx
            for adapter_idx, block_idx in enumerate(adapted_blocks)
        }
        self.film_adapters = nn.ModuleList(
            [FamilyFiLMAdapter(self.n_families, block_dims[block_idx]) for block_idx in adapted_blocks]
        )

        def _build_head(out_dim: int, *, zero_final: bool = False) -> nn.Module:
            head_layers: list[nn.Module] = []
            inner_prev = trunk_out
            for h in family_head_dims:
                head_layers.extend([nn.Linear(inner_prev, int(h)), make_activation(activation)])
                inner_prev = int(h)
            final = nn.Linear(inner_prev, int(out_dim))
            if zero_final:
                nn.init.zeros_(final.weight)
                nn.init.zeros_(final.bias)
            head_layers.append(final)
            return nn.Sequential(*head_layers)

        self.mu_shared_head = _build_head(1)
        self.delta_heads = nn.ModuleList(
            [_build_head(1, zero_final=True) for _ in range(self.n_families)]
        )
        self.log_var_head = _build_head(1)
        self.register_buffer(
            "trained_families",
            torch.zeros(self.n_families, dtype=torch.bool),
        )

    def mark_family_trained(self, family_id: int) -> None:
        idx = int(family_id)
        if 0 <= idx < self.n_families:
            self.trained_families[idx] = True

    def _validate_family_ids(self, family_id: torch.Tensor) -> None:
        if family_id.numel() == 0:
            return
        min_id = int(family_id.min().detach().cpu())
        max_id = int(family_id.max().detach().cpu())
        if min_id < 0 or max_id >= self.n_families:
            raise ValueError(
                f"family_id must be in [0, {self.n_families}); got min={min_id}, max={max_id}."
            )

    def _encode(self, features: torch.Tensor, family_id: torch.Tensor) -> torch.Tensor:
        h = features
        for idx, module in enumerate(self.trunk):
            h = module(h)
            adapter_idx = self._film_module_index_to_adapter.get(idx)
            if adapter_idx is not None:
                h = self.film_adapters[adapter_idx](h, family_id)
        return h

    def forward_parts(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        family_id = x[..., -1].long()
        self._validate_family_ids(family_id)
        features = x[..., :-1]
        h = self._encode(features, family_id)
        mu_shared = self.mu_shared_head(h)
        delta = torch.zeros(*h.shape[:-1], 1, device=h.device, dtype=h.dtype)
        for f in range(self.n_families):
            mask = family_id == f
            if mask.any():
                delta[mask] = self.delta_heads[f](h[mask])
        log_var = self.log_var_head(h)

        if self.training:
            unique = torch.unique(family_id).tolist()
            for f in unique:
                self.mark_family_trained(int(f))
        return mu_shared, delta, log_var

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu_shared, delta, log_var = self.forward_parts(x)
        return torch.cat([mu_shared + delta, log_var], dim=-1)

    def residual_delta_stats(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        _, delta, _ = self.forward_parts(x)
        delta_l2 = delta.pow(2).mean()
        return {
            "delta_l2": delta_l2,
            "delta_rms": torch.sqrt(delta_l2 + 1e-12),
        }

    def adapter_regularization_stats(self) -> dict[str, torch.Tensor]:
        l2_terms = [adapter.l2_from_identity() for adapter in self.film_adapters]
        film_l2 = torch.stack(l2_terms).mean()
        return {
            "film_l2": film_l2,
            "film_rms": torch.sqrt(film_l2 + 1e-12),
        }


def _normalize_film_adapter_placement(value: str) -> str:
    token = str(value or "last_block").strip().lower().replace("-", "_")
    aliases = {
        "last": "last_block",
        "final": "last_block",
        "final_block": "last_block",
        "last_block": "last_block",
        "all": "all_blocks",
        "all_block": "all_blocks",
        "all_blocks": "all_blocks",
    }
    if token not in aliases:
        raise ValueError(
            f"Unsupported film_adapter_placement={value!r}; expected last_block or all_blocks."
        )
    return aliases[token]


def single_state_dict_to_family_head(
    single_state: Mapping[str, torch.Tensor],
    family_state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Map a pooled two-output MLP into an exactly equivalent direct family head.

    The target must use ``family_head_dims=[]``. Its trunk receives the pooled
    trunk verbatim, the pooled mean row is duplicated into both family heads,
    and the pooled variance row becomes the shared variance head.
    """
    target = {key: value.detach().clone() for key, value in family_state.items()}
    output_weights = [
        key for key, value in single_state.items()
        if key.startswith("net.") and key.endswith(".weight") and value.ndim == 2 and value.shape[0] == 2
    ]
    if len(output_weights) != 1:
        raise ValueError(
            "Expected one pooled two-output Linear layer; "
            f"found {output_weights}."
        )
    source_weight_key = output_weights[0]
    source_bias_key = source_weight_key[:-len("weight")] + "bias"
    if source_bias_key not in single_state:
        raise KeyError(f"Missing pooled output bias: {source_bias_key}.")

    for key, value in single_state.items():
        if not key.startswith("net.") or key in {source_weight_key, source_bias_key}:
            continue
        mapped = "trunk." + key[len("net."):]
        if mapped in target and target[mapped].shape == value.shape:
            target[mapped] = value.detach().clone()

    source_weight = single_state[source_weight_key]
    source_bias = single_state[source_bias_key]
    head_mapping = {
        "mu_heads.0.0.weight": source_weight[0:1],
        "mu_heads.0.0.bias": source_bias[0:1],
        "mu_heads.1.0.weight": source_weight[0:1],
        "mu_heads.1.0.bias": source_bias[0:1],
        "log_var_head.0.weight": source_weight[1:2],
        "log_var_head.0.bias": source_bias[1:2],
    }
    missing = [key for key in head_mapping if key not in target]
    if missing:
        raise ValueError(
            "Target family head must use direct heads (family_head_dims=[]); "
            f"missing keys: {missing}."
        )
    for key, value in head_mapping.items():
        if target[key].shape != value.shape:
            raise ValueError(f"Shape mismatch while mapping {key}: {target[key].shape} != {value.shape}.")
        target[key] = value.detach().clone()
    if "trained_families" in target:
        target["trained_families"] = torch.ones_like(target["trained_families"], dtype=torch.bool)
    return target


def family_head_state_dict_to_residual(
    family_state: Mapping[str, torch.Tensor],
    residual_state: Mapping[str, torch.Tensor],
    *,
    shared_family_id: int = 1,
    preserve_family_offsets: bool = False,
) -> dict[str, torch.Tensor]:
    """Map a FamilyAwarePenetrationMLP state dict into residual-head layout.

    ``mu_heads[shared_family_id]`` becomes ``mu_shared_head``. Delta heads keep
    the target residual model's zero-initialised values so the converted model
    initially predicts the shared-family surface for every family. With
    ``preserve_family_offsets=True``, direct family heads are instead mapped to
    direct residual deltas, preserving both family predictions at step zero.
    """
    out = {key: value.detach().clone() for key, value in residual_state.items()}
    shared_prefix = f"mu_heads.{int(shared_family_id)}."
    for key, value in family_state.items():
        if key.startswith("trunk.") or key.startswith("log_var_head."):
            if key in out and out[key].shape == value.shape:
                out[key] = value.detach().clone()
            continue
        if key == "trained_families":
            if key in out:
                copied = out[key].detach().clone()
                n = min(int(copied.numel()), int(value.numel()))
                copied[:n] = value.detach().clone()[:n]
                out[key] = copied
            continue
        if key.startswith(shared_prefix):
            mapped = "mu_shared_head." + key[len(shared_prefix):]
            if mapped in out and out[mapped].shape == value.shape:
                out[mapped] = value.detach().clone()
    if preserve_family_offsets:
        if "trained_families" not in out:
            raise ValueError("Residual target is missing trained_families.")
        n_families = int(out["trained_families"].numel())
        required_source = {
            f"mu_heads.{family_id}.0.{suffix}"
            for family_id in range(n_families)
            for suffix in ("weight", "bias")
        }
        required_target = {
            f"delta_heads.{family_id}.0.{suffix}"
            for family_id in range(n_families)
            for suffix in ("weight", "bias")
        }
        missing = sorted(required_source - set(family_state)) + sorted(required_target - set(out))
        if missing:
            raise ValueError(
                "Function-preserving residual conversion requires direct family and delta heads "
                f"(family_head_dims=[]); missing keys: {missing}."
            )
        for family_id in range(n_families):
            for suffix in ("weight", "bias"):
                source_key = f"mu_heads.{family_id}.0.{suffix}"
                shared_key = f"mu_heads.{int(shared_family_id)}.0.{suffix}"
                target_key = f"delta_heads.{family_id}.0.{suffix}"
                if out[target_key].shape != family_state[source_key].shape:
                    raise ValueError(f"Shape mismatch while mapping {target_key}.")
                out[target_key] = (
                    family_state[source_key].detach().clone()
                    - family_state[shared_key].detach().clone()
                )
    return out


def state_dict_to_residual_film(
    source_state: Mapping[str, torch.Tensor],
    target_state: Mapping[str, torch.Tensor],
    *,
    source_architecture_mode: str,
    shared_family_id: int = 1,
) -> dict[str, torch.Tensor]:
    """Warm-start residual-FiLM from a family-head or residual-family model.

    The FiLM adapters remain at their target identity initialization unless the
    source is the exact same residual-FiLM layout. This preserves a step-0
    function match when inserting adapters into an existing production model.
    """
    source_arch = str(source_architecture_mode).lower()
    target = {key: value.detach().clone() for key, value in target_state.items()}
    if source_arch == "family_head":
        return family_head_state_dict_to_residual(
            source_state,
            target,
            shared_family_id=int(shared_family_id),
        )

    copy_prefixes = ("trunk.", "log_var_head.", "mu_shared_head.", "delta_heads.")
    if source_arch == "residual_family_head" or source_arch in RESIDUAL_FILM_ARCHITECTURE_MODES:
        for key, value in source_state.items():
            should_copy = key.startswith(copy_prefixes) or key == "trained_families"
            if should_copy and key in target and target[key].shape == value.shape:
                target[key] = value.detach().clone()
        return target

    raise ValueError(
        "Cannot warm-start residual-FiLM from architecture_mode="
        f"{source_architecture_mode!r}."
    )


def build_model(config: Mapping[str, Any]) -> nn.Module:
    architecture_mode = str(config.get("architecture_mode", "single")).lower()
    if architecture_mode == "family_head":
        return FamilyAwarePenetrationMLP(
            input_dim=int(config["input_dim"]),
            hidden_dims=[int(x) for x in config["hidden_dims"]],
            output_dim=int(config.get("output_dim", 2)),
            activation=str(config.get("activation", "gelu")),
            dropout=float(config.get("dropout", 0.0)),
            n_families=int(config.get("n_families", 2)),
            family_head_dims=[int(x) for x in config.get("family_head_dims", [128])],
            fallback_family_id=int(config.get("fallback_family_id", 1)),
        )
    if architecture_mode == "residual_family_head":
        return ResidualFamilyAwarePenetrationMLP(
            input_dim=int(config["input_dim"]),
            hidden_dims=[int(x) for x in config["hidden_dims"]],
            output_dim=int(config.get("output_dim", 2)),
            activation=str(config.get("activation", "gelu")),
            dropout=float(config.get("dropout", 0.0)),
            n_families=int(config.get("n_families", 2)),
            family_head_dims=[int(x) for x in config.get("family_head_dims", [128])],
            fallback_family_id=int(config.get("fallback_family_id", 1)),
        )
    if architecture_mode in RESIDUAL_FILM_ARCHITECTURE_MODES:
        if architecture_mode == "residual_film_last_block":
            film_adapter_placement = "last_block"
        elif architecture_mode == "residual_film_all_blocks":
            film_adapter_placement = "all_blocks"
        else:
            film_adapter_placement = str(config.get("film_adapter_placement", "last_block"))
        return ResidualFiLMFamilyAwarePenetrationMLP(
            input_dim=int(config["input_dim"]),
            hidden_dims=[int(x) for x in config["hidden_dims"]],
            output_dim=int(config.get("output_dim", 2)),
            activation=str(config.get("activation", "gelu")),
            dropout=float(config.get("dropout", 0.0)),
            n_families=int(config.get("n_families", 2)),
            family_head_dims=[int(x) for x in config.get("family_head_dims", [128])],
            fallback_family_id=int(config.get("fallback_family_id", 1)),
            film_adapter_placement=film_adapter_placement,
        )
    if architecture_mode != "single":
        raise ValueError(f"Unsupported architecture_mode={architecture_mode!r}")
    return PenetrationMLP(
        input_dim=int(config["input_dim"]),
        hidden_dims=[int(x) for x in config["hidden_dims"]],
        output_dim=int(config["output_dim"]),
        activation=str(config.get("activation", "gelu")),
        dropout=float(config.get("dropout", 0.0)),
        onset_aux_head=bool(config.get("onset_aux_head", False)),
        onset_aux_hidden=int(config.get("onset_aux_hidden", 64)),
        learnable_d2=bool(config.get("learnable_d2", False)),
        n_families_for_d2=int(config.get("n_families_for_d2", 6)),
        learnable_d2_floor=float(config.get("learnable_d2_floor", 1e-5)),
        learnable_d2_init=float(config.get("d2_concave_weight", 5e-4)),
    )


def sigmoid(x: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(x, dtype=float), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def spray_penetration_model_sigmoid(params: Sequence[np.ndarray | float], t: np.ndarray | float) -> np.ndarray:
    """Evaluate the blended physical penetration model at times t.

    Blends a √t (early ballistic) segment and a t^0.25 (Hiroyasu–Arai asymptote)
    segment via a logistic gate at transition time t0 with softness s.
    k_sqrt is negligible in practice; k_quarter is the dominant coefficient.
    """
    log_k_sqrt, log_k_quarter, log_t0, log_s = params
    k_sqrt = np.exp(log_k_sqrt)
    k_quarter = np.exp(log_k_quarter)
    t0 = np.exp(log_t0) + MIN_TIME_SHIFT_S
    s = np.exp(log_s)

    t_arr = np.clip(np.asarray(t, dtype=float), 1e-9, None)
    sqrt_segment = k_sqrt * np.sqrt(t_arr)
    quarter_segment = k_quarter * np.power(t_arr, 0.25)
    w = sigmoid((t_arr - t0) / np.maximum(s, 1e-12))
    return (1.0 - w) * sqrt_segment + w * quarter_segment


def spray_penetration_model_sigmoid_d_dt(params: Sequence[np.ndarray | float], t: np.ndarray | float) -> np.ndarray:
    log_k_sqrt, log_k_quarter, log_t0, log_s = params
    k_sqrt = np.exp(log_k_sqrt)
    k_quarter = np.exp(log_k_quarter)
    t0 = np.exp(log_t0) + MIN_TIME_SHIFT_S
    s = np.exp(log_s)

    t_arr = np.clip(np.asarray(t, dtype=float), 1e-9, None)
    sqrt_segment = k_sqrt * np.sqrt(t_arr)
    quarter_segment = k_quarter * np.power(t_arr, 0.25)
    w = sigmoid((t_arr - t0) / np.maximum(s, 1e-12))
    dw_dt = w * (1.0 - w) / np.maximum(s, 1e-12)
    d_sqrt_dt = 0.5 * k_sqrt / np.sqrt(t_arr)
    d_quarter_dt = 0.25 * k_quarter * np.power(t_arr, -0.75)
    return (1.0 - w) * d_sqrt_dt + w * d_quarter_dt + dw_dt * (quarter_segment - sqrt_segment)


def reconstruct_penetration_series(
    log_k_sqrt: float,
    log_k_quarter: float,
    log_t0: float,
    log_s: float,
    time_s: np.ndarray,
) -> np.ndarray:
    return spray_penetration_model_sigmoid([log_k_sqrt, log_k_quarter, log_t0, log_s], time_s)
