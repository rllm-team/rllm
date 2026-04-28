#  Copyright (c) Prior Labs GmbH 2025.

# TODO: This module needs some tidying
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal
from typing_extensions import override

import torch
from torch import nn

from rllm.nn.loss.base_loss import BaseLoss

if TYPE_CHECKING:
    from matplotlib import pyplot as plt

Reduction = Literal["none", "mean", "sum"]


def _reduce_loss(loss: torch.Tensor, reduction: Reduction) -> torch.Tensor:
    if reduction == "none":
        return loss
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    raise ValueError(f"Unsupported reduction: {reduction}")


class BarDistribution(BaseLoss):
    r"""Negative log-density loss for piecewise-constant regression outputs.

    The model predicts logits over adjacent intervals defined by ``borders``.
    After softmax, each bucket probability is divided by the bucket width so the
    loss is a density, not a plain class cross entropy.  The default
    ``reduction="none"`` is intentionally compatible with the TabPFN runtime,
    which aggregates losses outside of this module.

    Args:
        borders: Sorted 1-D tensor of bucket boundaries.
        ignore_nan_targets: If ``True``, NaN targets contribute zero loss.
            If ``False``, NaN targets raise ``ValueError``.
        reduction: ``"none"``, ``"mean"``, or ``"sum"``.
    """

    def __init__(
        self,
        borders: torch.Tensor,
        *,
        ignore_nan_targets: bool = True,
        reduction: Reduction = "none",
    ) -> None:
        super().__init__()
        if borders.ndim != 1:
            raise ValueError(f"borders must be 1-D, got shape {tuple(borders.shape)}")
        if borders.numel() < 2:
            raise ValueError("borders must contain at least two values.")

        borders = borders.contiguous()
        self.register_buffer("borders", borders)
        full_width = self.bucket_widths.sum()

        support_width = self.borders[-1] - self.borders[0]
        if not torch.isclose(full_width, support_width, rtol=1e-2, atol=1e-6):
            raise ValueError(
                "Bucket widths do not cover the support: "
                f"diff={full_width - support_width}, "
                f"full_width={full_width}, "
                f"left={self.borders[0]}, right={self.borders[-1]}"
            )

        # This also allows size zero buckets
        if not (self.bucket_widths >= 0.0).all():
            raise ValueError("Please provide sorted borders.")
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(f"Unsupported reduction: {reduction}")

        self.ignore_nan_targets = ignore_nan_targets
        self.reduction = reduction

    @property
    def bucket_widths(self) -> torch.Tensor:
        return self.borders[1:] - self.borders[:-1]

    @property
    def num_bars(self) -> int:
        return len(self.borders) - 1

    def _check_logits(self, logits: torch.Tensor) -> None:
        if logits.shape[-1] != self.num_bars:
            raise ValueError(
                f"Expected logits last dimension to be {self.num_bars}, "
                f"got {logits.shape[-1]}."
            )

    def cdf(self, logits: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
        """Calculates the cdf of the distribution described by the logits.
        The cdf is scaled by the width of the bars.

        Args:
            logits:
                tensor of shape (batch_size, ..., num_bars) with the logits describing
                the distribution
            ys:
                tensor of shape (batch_size, ..., n_ys to eval) or (n_ys to eval)
                with the targets.
        """
        self._check_logits(logits)
        ys = ys.to(device=logits.device, dtype=logits.dtype)
        if len(ys.shape) < len(logits.shape) and len(ys.shape) == 1:
            # bring new borders to the same dim as logits up to the last dim
            ys = ys.repeat(logits.shape[:-1] + (1,))
        else:
            if ys.shape[:-1] != logits.shape[:-1]:
                raise ValueError(
                    f"ys.shape: {ys.shape} logits.shape: {logits.shape}"
                )
        probs = torch.softmax(logits, dim=-1)
        buckets_of_ys = self.map_to_bucket_idx(ys).clamp(0, self.num_bars - 1)
        borders = self.borders.to(device=logits.device, dtype=logits.dtype)
        bucket_widths = self.bucket_widths.to(device=logits.device, dtype=logits.dtype)

        prob_so_far = torch.cumsum(probs, dim=-1) - probs
        prob_left_of_bucket = prob_so_far.gather(-1, buckets_of_ys)

        share_of_bucket_left = (
            (ys - borders[buckets_of_ys]) / bucket_widths[buckets_of_ys]
        ).clamp(0.0, 1.0)
        prob_in_bucket = probs.gather(-1, buckets_of_ys) * share_of_bucket_left

        prob_left_of_ys = prob_left_of_bucket + prob_in_bucket

        # just to fix numerical inaccuracies, if we had *exact* computation above we
        # would not need the following:
        prob_left_of_ys[ys <= borders[0]] = 0.0
        prob_left_of_ys[ys >= borders[-1]] = 1.0
        assert not torch.isnan(prob_left_of_ys).any()

        return prob_left_of_ys.clip(0.0, 1.0)

    def get_probs_for_different_borders(
        self,
        logits: torch.Tensor,
        new_borders: torch.Tensor,
    ) -> torch.Tensor:
        """The logits describe the density of the distribution over the current
        self.borders.

        This function returns the logits if the self.borders were changed
        to new_borders. This is useful to average the logits of different models.
        """
        new_borders = new_borders.to(device=logits.device, dtype=logits.dtype)
        current_borders = self.borders.to(device=logits.device, dtype=logits.dtype)
        if (len(current_borders) == len(new_borders)) and torch.equal(
            current_borders,
            new_borders,
        ):
            return logits.softmax(-1)

        prob_left_of_borders = self.cdf(logits, new_borders)
        prob_left_of_borders[..., 0] = 0.0
        prob_left_of_borders[..., -1] = 1.0

        return (
            prob_left_of_borders[..., 1:] - prob_left_of_borders[..., :-1]
        ).clamp_min(0.0)

    def average_bar_distributions_into_this(
        self,
        list_of_bar_distributions: Sequence[BarDistribution],
        list_of_logits: Sequence[torch.Tensor],
        *,
        average_logits: bool = False,
    ) -> torch.Tensor:
        """:param list_of_bar_distributions:
        :param list_of_logits:
        :param average_logits:
        :return:
        """
        probs = torch.stack(
            [
                bar_dist.get_probs_for_different_borders(logits, self.borders)
                for bar_dist, logits in zip(list_of_bar_distributions, list_of_logits)
            ],
            dim=0,
        )

        if average_logits:
            probs = probs.log().mean(dim=0).softmax(-1)
        else:
            probs = probs.mean(dim=0)

        return probs.log()

    def __setstate__(self, state: dict) -> None:
        if "bucket_widths" in state:
            del state["bucket_widths"]
        super().__setstate__(state)
        self.__dict__.setdefault("append_mean_pred", False)
        self.__dict__.setdefault("ignore_nan_targets", True)
        self.__dict__.setdefault("reduction", "none")

    def map_to_bucket_idx(self, y: torch.Tensor) -> torch.Tensor:
        # assert the borders are actually sorted
        borders = self.borders.to(device=y.device, dtype=y.dtype)
        if not (borders[1:] - borders[:-1] >= 0.0).all():
            raise ValueError("Please provide sorted borders.")
        target_sample = torch.searchsorted(borders, y) - 1
        target_sample[y == borders[0]] = 0
        target_sample[y == borders[-1]] = self.num_bars - 1
        return target_sample

    def ignore_init(self, y: torch.Tensor) -> torch.Tensor:
        ignore_loss_mask = torch.isnan(y)
        if ignore_loss_mask.any() and not self.ignore_nan_targets:
            raise ValueError(f"Found NaN in target {y}")

        # this is just a default value, it will be ignored anyway
        y[ignore_loss_mask] = self.borders.to(device=y.device, dtype=y.dtype)[0]
        return ignore_loss_mask

    def compute_scaled_log_probs(self, logits: torch.Tensor) -> torch.Tensor:
        # this is equivalent to log(p(y)) of the density p
        bucket_log_probs = torch.log_softmax(logits, -1)
        bucket_widths = self.bucket_widths.to(device=logits.device, dtype=logits.dtype)
        return bucket_log_probs - torch.log(bucket_widths)

    def full_ce(self, logits: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        return -(probs * torch.log_softmax(logits, -1)).sum(-1)

    def forward(
        self,
        logits: torch.Tensor,
        y: torch.Tensor,
        mean_prediction_logits: torch.Tensor | Any | None = None,
        *,
        cat_ix: Any | None = None,
    ) -> torch.Tensor:
        # gives the negative log density (the _loss_),
        # y: T x B, logits: T x B x self.num_bars
        # Backward compatibility:
        # - Legacy TabPFN paths may pass `cat_ix[1]` as the third positional arg.
        # - Historical BO code may pass `mean_prediction_logits` as the third arg.
        if mean_prediction_logits is not None and not torch.is_tensor(
            mean_prediction_logits
        ):
            if cat_ix is not None:
                raise ValueError(
                    "Received both positional and keyword cat_ix. "
                    "Please pass cat_ix only once."
                )
            cat_ix = mean_prediction_logits
            mean_prediction_logits = None
        # Keep this argument for caller compatibility with generic loss paths.
        del cat_ix
        self._check_logits(logits)
        y = y.to(device=logits.device, dtype=logits.dtype).clone().view(
            *logits.shape[:-1],
        )  # no trailing one dimension
        ignore_loss_mask = self.ignore_init(y)
        target_sample = self.map_to_bucket_idx(y)
        if not (target_sample >= 0).all() or not (target_sample < self.num_bars).all():
            raise ValueError(
                f"y {y} not in support set for borders (min_y, max_y) {self.borders}"
            )

        scaled_bucket_log_probs = self.compute_scaled_log_probs(logits)

        # T x B
        nll_loss = -scaled_bucket_log_probs.gather(
            -1,
            target_sample[..., None],
        ).squeeze(-1)
        nll_loss[ignore_loss_mask] = 0.0

        # TODO(eddiebergman): Verify if this is still relevant
        # TO BE REMOVED AFTER BO SUBMISSION
        if mean_prediction_logits is not None:
            nll_loss = torch.cat(
                (nll_loss, self.mean_loss(logits, mean_prediction_logits)),
                0,
            )

        return _reduce_loss(nll_loss, self.reduction)

    # TODO(eddiebergman): Verify if this is still relevant
    def mean_loss(
        self,
        logits: torch.Tensor,
        mean_prediction_logits: torch.Tensor,
    ) -> torch.Tensor:  # TO BE REMOVED AFTER BO SUBMISSION
        scaled_mean_log_probs = self.compute_scaled_log_probs(mean_prediction_logits)
        if not self.training:
            pass
        assert len(logits.shape) == 3, len(logits.shape)
        assert len(scaled_mean_log_probs.shape) == 2, len(scaled_mean_log_probs.shape)

        means = self.mean(logits).detach()  # T x B

        # T x B
        target_mean = self.map_to_bucket_idx(means).clamp_(0, self.num_bars - 1)

        # 1 x B
        return -scaled_mean_log_probs.gather(1, target_mean.T).mean(1).unsqueeze(0)

    def mean(self, logits: torch.Tensor) -> torch.Tensor:
        borders = self.borders.to(device=logits.device, dtype=logits.dtype)
        bucket_widths = self.bucket_widths.to(device=logits.device, dtype=logits.dtype)
        bucket_means = borders[:-1] + bucket_widths / 2
        p = torch.softmax(logits, -1)
        return p @ bucket_means

    def median(self, logits: torch.Tensor) -> torch.Tensor:
        return self.icdf(logits, 0.5)

    # TODO(eddiebergman): Check if still relevant
    def cdf_temporary(self, logits: torch.Tensor) -> torch.Tensor:
        """Cumulative distribution function.

        TODO: this already exists here, make sure to merge, at the moment still used.
        """
        probs = logits.softmax(-1)
        return -torch.cumsum(probs, -1).sum(axis=-1)  # type: ignore

    def icdf(
        self,
        logits: torch.Tensor,
        left_prob: float | torch.Tensor,
    ) -> torch.Tensor:
        """Implementation of the quantile function
        :param logits: Tensor of any shape, with the last dimension being logits
        :param left_prob: Probability mass to the left of the result. It can be
            a scalar or a tensor matching ``logits.shape[:-1]``.
        :return: Position with `left_prob` probability weight to the left.
        """
        self._check_logits(logits)
        probs = logits.softmax(-1)
        cumprobs = torch.cumsum(probs, -1)
        left_prob_tensor = torch.as_tensor(
            left_prob,
            device=logits.device,
            dtype=logits.dtype,
        )
        if left_prob_tensor.ndim == 0:
            left_prob_tensor = left_prob_tensor.expand(cumprobs.shape[:-1])
        elif left_prob_tensor.shape != cumprobs.shape[:-1]:
            raise ValueError(
                "left_prob must be a scalar or match logits.shape[:-1], got "
                f"{tuple(left_prob_tensor.shape)} for logits shape {tuple(logits.shape)}"
            )
        idx = (
            torch.searchsorted(
                cumprobs,
                left_prob_tensor.unsqueeze(-1),
            )
            .squeeze(-1)
            .clamp(0, cumprobs.shape[-1] - 1)
        )  # this might not do the right for outliers
        cumprobs = torch.cat(
            [torch.zeros(*cumprobs.shape[:-1], 1, device=logits.device), cumprobs],
            -1,
        )

        rest_prob = left_prob_tensor - cumprobs.gather(-1, idx[..., None]).squeeze(-1)
        borders = self.borders.to(device=logits.device, dtype=logits.dtype)
        left_border = borders[idx]
        right_border = borders[idx + 1]
        return left_border + (right_border - left_border) * rest_prob / probs.gather(
            -1,
            idx[..., None],
        ).squeeze(-1).clamp_min(torch.finfo(logits.dtype).tiny)

    def quantile(
        self,
        logits: torch.Tensor,
        center_prob: float = 0.682,
    ) -> torch.Tensor:
        side_probs = (1.0 - center_prob) / 2
        return torch.stack(
            (self.icdf(logits, side_probs), self.icdf(logits, 1.0 - side_probs)),
            -1,
        )

    def ucb(
        self,
        logits: torch.Tensor,
        best_f: float,  # noqa: ARG002
        rest_prob: float = (1 - 0.682) / 2,
        *,
        maximize: bool = True,
    ) -> torch.Tensor:
        """UCB utility. Rest Prob is the amount of utility above (below) the confidence
        interval that is ignored.

        Higher rest_prob is equivalent to lower beta in the standard GP-UCB formulation.

        Args:
            logits: Logits, as returned by the Transformer.
            rest_prob:
                The amount of utility above (below) the confidence interval that is
                ignored.

                The default is equivalent to using GP-UCB with `beta=1`.
                To get the corresponding `beta`, where `beta` is from
                the standard GP definition of UCB `ucb_utility = mean + beta * std`,
                you can use this computation:

                `beta = math.sqrt(2)*torch.erfinv(torch.tensor(2*(1-rest_prob)-1))`
            best_f: Unused
            maximize: Whether to maximize.
        """
        if maximize:
            rest_prob = 1 - rest_prob
        return self.icdf(logits, rest_prob)

    def mode(self, logits: torch.Tensor) -> torch.Tensor:
        bucket_widths = self.bucket_widths.to(device=logits.device, dtype=logits.dtype)
        density = logits.softmax(-1) / bucket_widths
        mode_inds = density.argmax(-1)
        borders = self.borders.to(device=logits.device, dtype=logits.dtype)
        bucket_means = borders[:-1] + bucket_widths / 2
        return bucket_means[mode_inds]

    def ei(
        self,
        logits: torch.Tensor,
        best_f: float | torch.Tensor,
        *,
        maximize: bool = True,
    ) -> torch.Tensor:  # logits: evaluation_points x batch x feature_dim
        borders = self.borders.to(device=logits.device, dtype=logits.dtype)
        bucket_diffs = borders[1:] - borders[:-1]
        if not maximize:
            raise NotImplementedError("Minimization EI is not implemented.")
        if not torch.is_tensor(best_f) or not len(best_f.shape):  # type: ignore
            best_f = torch.full(  # type: ignore
                logits[..., 0].shape,
                best_f,
                device=logits.device,
                dtype=logits.dtype,
            )
        else:
            best_f = best_f.to(device=logits.device, dtype=logits.dtype)

        best_f = best_f[..., None].repeat(*[1] * len(best_f.shape), logits.shape[-1])  # type: ignore
        clamped_best_f = best_f.clamp(borders[:-1], borders[1:])

        # > bucket_contributions =
        # >     (best_f[...,None] < self.borders[:-1]).float() * bucket_means
        # true bucket contributions
        bucket_contributions = (
            (borders[1:] ** 2 - clamped_best_f**2) / 2
            - best_f * (borders[1:] - clamped_best_f)
        ) / bucket_diffs

        p = torch.softmax(logits, -1)
        return torch.einsum("...b,...b->...", p, bucket_contributions)

    def pi(
        self,
        logits: torch.Tensor,
        best_f: float | torch.Tensor,
        *,
        maximize: bool = True,
    ) -> torch.Tensor:  # logits: evaluation_points x batch x feature_dim
        """Acquisition Function: Probability of Improvement.

        Args:
            logits: as returned by Transformer
            best_f: best evaluation so far (the incumbent)
            maximize: whether to maximize

        Returns:
            probability of improvement
        """
        if not maximize:
            raise NotImplementedError("Minimization PI is not implemented.")
        if not torch.is_tensor(best_f) or not len(best_f.shape):  # type: ignore
            best_f = torch.full(  # type: ignore
                logits[..., 0].shape,
                best_f,
                device=logits.device,
                dtype=logits.dtype,
            )
        else:
            best_f = best_f.to(device=logits.device, dtype=logits.dtype)
        p = torch.softmax(logits, -1)
        borders = self.borders.to(device=logits.device, dtype=logits.dtype)
        border_widths = borders[1:] - borders[:-1]
        factor = 1.0 - ((best_f[..., None] - borders[:-1]) / border_widths).clamp(  # type: ignore
            0.0,
            1.0,
        )
        return (p * factor).sum(-1)

    def mean_of_square(self, logits: torch.Tensor) -> torch.Tensor:
        """Computes E[x^2].

        Args:
            logits: Output of the model.

        Returns:
            mean of square
        """
        left_borders = self.borders[:-1].to(device=logits.device, dtype=logits.dtype)
        right_borders = self.borders[1:].to(device=logits.device, dtype=logits.dtype)
        bucket_mean_of_square = (
            left_borders.square()
            + right_borders.square()
            + left_borders * right_borders
        ) / 3.0
        p = torch.softmax(logits, -1)
        return p @ bucket_mean_of_square

    def variance(self, logits: torch.Tensor) -> torch.Tensor:
        return self.mean_of_square(logits) - self.mean(logits).square()

    # TODO: Move into standalone module for plotting
    def plot(
        self,
        logits: torch.Tensor,
        ax: plt.Axes | None = None,
        zoom_to_quantile: float | None = None,
        **kwargs: Any,
    ) -> plt.Axes:
        """Plots the distribution."""
        import matplotlib.pyplot as plt

        logits = logits.squeeze()
        assert logits.dim() == 1, "logits should be 1d, at least after squeezing."
        if ax is None:
            ax = plt.gca()
        if zoom_to_quantile is not None:
            lower_bounds, upper_bounds = self.quantile(
                logits,
                zoom_to_quantile,
            ).transpose(0, -1)
            lower_bound = lower_bounds.min().item()
            upper_bound = upper_bounds.max().item()
            ax.set_xlim(lower_bound, upper_bound)
            border_mask = (self.borders[:-1] >= lower_bound) & (
                self.borders[1:] <= upper_bound
            )
        else:
            border_mask = slice(None)
        p = torch.softmax(logits, -1) / self.bucket_widths
        ax.bar(
            self.borders[:-1][border_mask],
            p[border_mask],
            self.bucket_widths[border_mask],
            **kwargs,
        )
        return ax


class FullSupportBarDistribution(BarDistribution):
    r"""Bar distribution with half-normal tails on both support edges.

    This is the regression criterion used by the retained TabPFN runtime.  It
    behaves like :class:`BarDistribution` for interior buckets and extends the
    first and last buckets with half-normal tail densities.
    """

    def __init__(
        self,
        borders: torch.Tensor,
        **kwargs: Any,
    ) -> None:
        super().__init__(borders, **kwargs)
        self.assert_support(allow_zero_bucket_left=False)

        losses_per_bucket = torch.zeros_like(self.bucket_widths)
        self.register_buffer("losses_per_bucket", losses_per_bucket)

    def assert_support(self, *, allow_zero_bucket_left: bool = False) -> None:
        if allow_zero_bucket_left:
            if self.bucket_widths[-1] <= 0:
                raise ValueError(
                    "Half Normal weight must be > 0 "
                    f"(got -1:{self.bucket_widths[-1]})."
                )
            # This fixes the distribution if the half normal at zero is width zero
            if self.bucket_widths[0] == 0:
                self.borders[0] = self.borders[0] - 1
        else:
            if self.bucket_widths[0] <= 0 or self.bucket_widths[-1] <= 0:
                raise ValueError(
                    "FullSupportBarDistribution requires positive edge buckets."
                )

    @staticmethod
    def halfnormal_with_p_weight_before(
        range_max: float | torch.Tensor,
        p: float = 0.5,
    ) -> torch.distributions.HalfNormal:
        range_tensor = torch.as_tensor(range_max)
        unit_scale = torch.ones(
            (),
            dtype=range_tensor.dtype,
            device=range_tensor.device,
        )
        prob = torch.as_tensor(p, dtype=range_tensor.dtype, device=range_tensor.device)
        s = range_tensor / torch.distributions.HalfNormal(unit_scale).icdf(
            prob,
        )
        return torch.distributions.HalfNormal(s)

    @override
    def forward(
        self,
        logits: torch.Tensor,
        y: torch.Tensor,
        mean_prediction_logits: torch.Tensor | Any | None = None,
        *,
        cat_ix: Any | None = None,
    ) -> torch.Tensor:
        """Returns the negative log density (the _loss_).

        y: T x B, logits: T x B x self.num_bars.

        :param logits: Tensor of shape T x B x self.num_bars
        :param y: Tensor of shape T x B
        :param mean_prediction_logits:
        :param cat_ix: Optional categorical index placeholder for caller
            compatibility with generic loss invocation paths.
        :return:
        """
        if mean_prediction_logits is not None and not torch.is_tensor(
            mean_prediction_logits
        ):
            if cat_ix is not None:
                raise ValueError(
                    "Received both positional and keyword cat_ix. "
                    "Please pass cat_ix only once."
                )
            cat_ix = mean_prediction_logits
            mean_prediction_logits = None
        del cat_ix
        if self.num_bars <= 1:
            raise ValueError("FullSupportBarDistribution requires at least two bars.")
        self._check_logits(logits)
        y = y.to(device=logits.device, dtype=logits.dtype).clone().view(
            *logits.shape[:-1],
        )  # no trailing one dimension
        ignore_loss_mask = self.ignore_init(y)  # alters y
        target_sample = self.map_to_bucket_idx(y)  # shape: T x B (same as y)
        target_sample.clamp_(0, self.num_bars - 1)

        if not (target_sample >= 0).all() or not (target_sample < self.num_bars).all():
            raise ValueError(
                f"y {y} not in support set for borders (min_y, max_y) {self.borders}"
            )
        # ignore all position with nan values

        scaled_bucket_log_probs = self.compute_scaled_log_probs(logits)

        if len(scaled_bucket_log_probs) != len(target_sample):
            raise ValueError(
                f"scaled log probs shape {scaled_bucket_log_probs.shape} is not "
                f"compatible with target shape {target_sample.shape}"
            )
        log_probs = scaled_bucket_log_probs.gather(
            -1,
            target_sample.unsqueeze(-1),
        ).squeeze(-1)

        borders = self.borders.to(device=logits.device, dtype=logits.dtype)
        bucket_widths = self.bucket_widths.to(device=logits.device, dtype=logits.dtype)
        side_normals = (
            self.halfnormal_with_p_weight_before(bucket_widths[0]),
            self.halfnormal_with_p_weight_before(bucket_widths[-1]),
        )

        log_probs[target_sample == 0] += side_normals[0].log_prob(
            (borders[1] - y[target_sample == 0]).clamp(min=0.00000001),
        ) + torch.log(bucket_widths[0])
        log_probs[target_sample == self.num_bars - 1] += side_normals[1].log_prob(
            (y[target_sample == self.num_bars - 1] - borders[-2]).clamp(
                min=0.00000001,
            ),
        ) + torch.log(bucket_widths[-1])

        nll_loss = -log_probs
        if ignore_loss_mask.any():
            nll_loss[ignore_loss_mask] = 0.0
        nll_loss_for_stats = nll_loss

        if mean_prediction_logits is not None:  # TO BE REMOVED AFTER BO PAPER IS DONE
            assert (
                not ignore_loss_mask.any()
            ), "Ignoring examples is not implemented with mean pred."
            if not torch.is_grad_enabled():
                pass
            nll_loss = torch.cat(
                (nll_loss, self.mean_loss(logits, mean_prediction_logits)),
                0,
            )

        # TODO: Check with samuel whether to keep
        valid_target_sample = target_sample[~ignore_loss_mask].flatten()
        valid_nll_loss = nll_loss_for_stats[~ignore_loss_mask].flatten().detach()
        if valid_target_sample.numel() > 0:
            bucket_updates = torch.zeros(
                self.num_bars,
                device=valid_nll_loss.device,
                dtype=self.losses_per_bucket.dtype,
            )
            bucket_updates.scatter_add_(
                0,
                valid_target_sample.to(bucket_updates.device),
                valid_nll_loss.to(bucket_updates.dtype),
            )
            self.losses_per_bucket += (
                bucket_updates.to(self.losses_per_bucket.device)
                / valid_target_sample.numel()
            )

        return _reduce_loss(nll_loss, self.reduction)

    def pdf(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Probability density function at y."""
        return torch.exp(-self.forward(logits, y))

    def sample(self, logits: torch.Tensor, t: float = 1.0) -> torch.Tensor:
        """Samples values from the distribution.

        Temperature t.
        """
        p_cdf = torch.rand(
            logits.shape[:-1],
            device=logits.device,
            dtype=logits.dtype,
        )
        return self.icdf(logits / t, p_cdf)

    @override
    def mean(self, logits: torch.Tensor) -> torch.Tensor:
        borders = self.borders.to(device=logits.device, dtype=logits.dtype)
        bucket_widths = self.bucket_widths.to(device=logits.device, dtype=logits.dtype)
        bucket_means = borders[:-1] + bucket_widths / 2
        p = torch.softmax(logits, -1)
        side_normals = (
            self.halfnormal_with_p_weight_before(bucket_widths[0]),
            self.halfnormal_with_p_weight_before(bucket_widths[-1]),
        )
        bucket_means[0] = -side_normals[0].mean + borders[1]
        bucket_means[-1] = side_normals[1].mean + borders[-2]
        return p @ bucket_means

    @override
    def mean_of_square(self, logits: torch.Tensor) -> torch.Tensor:
        """Computes E[x^2].

        Args:
            logits: Output of the model.
        """
        borders = self.borders.to(device=logits.device, dtype=logits.dtype)
        bucket_widths = self.bucket_widths.to(device=logits.device, dtype=logits.dtype)
        left_borders = borders[:-1]
        right_borders = borders[1:]
        bucket_mean_of_square = (
            left_borders.square()
            + right_borders.square()
            + left_borders * right_borders
        ) / 3.0
        side_normals = (
            self.halfnormal_with_p_weight_before(bucket_widths[0]),
            self.halfnormal_with_p_weight_before(bucket_widths[-1]),
        )
        bucket_mean_of_square[0] = (
            side_normals[0].variance
            + (-side_normals[0].mean + borders[1]).square()
        )
        bucket_mean_of_square[-1] = (
            side_normals[1].variance
            + (side_normals[1].mean + borders[-2]).square()
        )
        p = torch.softmax(logits, -1)
        return p @ bucket_mean_of_square

    @override
    def pi(
        self,
        logits: torch.Tensor,
        best_f: torch.Tensor | float,
        *,
        maximize: bool = True,
    ) -> torch.Tensor:
        """Acquisition Function: Probability of Improvement.

        Args:
            logits: as returned by Transformer (evaluation_points x batch x feature_dim)
            best_f: best evaluation so far (the incumbent)
            maximize: whether to maximize
        """
        # logits: evaluation_points x batch x feature_dim
        if not maximize:
            raise NotImplementedError("Minimization PI is not implemented.")
        if not torch.is_tensor(best_f) or not len(best_f.shape):  # type: ignore
            # evaluation_points x batch
            best_f = torch.full(  # type: ignore
                logits[..., 0].shape,
                best_f,
                device=logits.device,
                dtype=logits.dtype,
            )
        else:
            best_f = best_f.to(device=logits.device, dtype=logits.dtype)

        if best_f.shape != logits[..., 0].shape:  # type: ignore
            raise ValueError(
                f"best_f.shape: {best_f.shape}, logits.shape: {logits.shape}"  # type: ignore
            )
        p = torch.softmax(logits, -1)  # evaluation_points x batch
        borders = self.borders.to(device=logits.device, dtype=logits.dtype)
        bucket_widths = self.bucket_widths.to(device=logits.device, dtype=logits.dtype)
        factor = 1.0 - ((best_f[..., None] - borders[:-1]) / bucket_widths).clamp(  # type: ignore
            0.0,
            1.0,
        )  # evaluation_points x batch x num_bars

        side_normals = (
            self.halfnormal_with_p_weight_before(bucket_widths[0]),
            self.halfnormal_with_p_weight_before(bucket_widths[-1]),
        )
        position_in_side_normals = (
            -(best_f - borders[1]).clamp(max=0.0),
            (best_f - borders[-2]).clamp(min=0.0),
        )  # evaluation_points x batch
        factor[..., 0] = 0.0
        factor[..., 0][position_in_side_normals[0] > 0.0] = side_normals[0].cdf(
            position_in_side_normals[0][position_in_side_normals[0] > 0.0],
        )
        factor[..., -1] = 1.0
        factor[..., -1][position_in_side_normals[1] > 0.0] = 1.0 - side_normals[1].cdf(
            position_in_side_normals[1][position_in_side_normals[1] > 0.0],
        )
        return (p * factor).sum(-1)

    def ei_for_halfnormal(
        self,
        scale: float | torch.Tensor,
        best_f: torch.Tensor | float,
        *,
        maximize: bool = True,
    ) -> torch.Tensor:
        """EI for a standard normal distribution with mean 0 and variance `scale` times 2.

        This is the same as the half-normal EI and was checked against a
        Monte Carlo approximation in the upstream TabPFN implementation.
        """
        if not maximize:
            raise NotImplementedError("Minimization EI is not implemented.")
        if torch.is_tensor(best_f):
            scale = torch.as_tensor(
                scale,
                device=best_f.device,
                dtype=best_f.dtype,
            )
            best_f = best_f.to(device=scale.device, dtype=scale.dtype)
        else:
            scale = torch.as_tensor(scale)
            best_f = torch.as_tensor(best_f, device=scale.device, dtype=scale.dtype)
        mean = torch.zeros((), device=best_f.device, dtype=best_f.dtype)
        u = (mean - best_f) / scale
        normal = torch.distributions.Normal(torch.zeros_like(u), torch.ones_like(u))
        ucdf = normal.cdf(u)
        updf = torch.exp(normal.log_prob(u))
        normal_ei = scale * (updf + u * ucdf)
        return 2 * normal_ei

    @override
    def ei(
        self,
        logits: torch.Tensor,
        best_f: torch.Tensor | float,
        *,
        maximize: bool = True,
    ) -> torch.Tensor:
        # logits: evaluation_points x batch x feature_dim
        if torch.isnan(logits).any():
            raise ValueError(f"logits contains NaNs: {logits}")
        if not maximize:
            raise NotImplementedError("Minimization EI is not implemented.")
        borders = self.borders.to(device=logits.device, dtype=logits.dtype)
        bucket_diffs = self.bucket_widths.to(device=logits.device, dtype=logits.dtype)
        if not torch.is_tensor(best_f) or not len(best_f.shape):  # type: ignore
            best_f = torch.full(  # type: ignore
                logits[..., 0].shape,
                best_f,
                device=logits.device,
                dtype=logits.dtype,
            )
        else:
            best_f = best_f.to(device=logits.device, dtype=logits.dtype)

        if best_f.shape != logits[..., 0].shape:  # type: ignore
            raise ValueError(
                f"best_f.shape: {best_f.shape}, logits.shape: {logits.shape}"  # type: ignore
            )

        best_f_per_logit = best_f[..., None].repeat(  # type: ignore
            *[1] * len(best_f.shape),  # type: ignore
            logits.shape[-1],
        )
        clamped_best_f = best_f_per_logit.clamp(borders[:-1], borders[1:])

        # true bucket contributions
        bucket_contributions = (
            (borders[1:] ** 2 - clamped_best_f**2) / 2
            - best_f_per_logit * (borders[1:] - clamped_best_f)
        ) / bucket_diffs

        # extra stuff for continuous
        side_normals = (
            self.halfnormal_with_p_weight_before(bucket_diffs[0]),
            self.halfnormal_with_p_weight_before(bucket_diffs[-1]),
        )
        position_in_side_normals = (
            -(best_f - borders[1]).clamp(max=0.0),
            (best_f - borders[-2]).clamp(min=0.0),
        )  # evaluation_points x batch

        bucket_contributions[..., -1] = self.ei_for_halfnormal(
            side_normals[1].scale,
            position_in_side_normals[1],
        )

        bucket_contributions[..., 0] = self.ei_for_halfnormal(
            side_normals[0].scale,
            torch.zeros_like(position_in_side_normals[0]),
        ) - self.ei_for_halfnormal(side_normals[0].scale, position_in_side_normals[0])

        p = torch.softmax(logits, -1)
        return torch.einsum("...b,...b->...", p, bucket_contributions)


def get_bucket_limits(
    num_outputs: int,
    full_range: tuple | None = None,
    ys: torch.Tensor | None = None,
    *,
    verbose: bool = False,  # noqa: ARG001
    widen_bucket_limits_factor: float | None = None,
) -> torch.Tensor:
    """Create bucket limits from either a fixed range or empirical targets.

    Args:
        num_outputs:
            This is only tested for num_outputs=1, but should work for larger
            num_outputs as well.
        full_range:
            If ys is not passed, this is the range of the ys that should be
            used to estimate the bucket limits.
        ys:
            If ys is passed, this is the ys that should be used to estimate the bucket
            limits. Do not pass full_range in this case.
        verbose: Unused
        widen_bucket_limits_factor:
            If set, the bucket limits are widened by this factor.
            This allows to have a slightly larger range than the actual data.
    """
    if (ys is None) == (full_range is None):
        raise ValueError("Either full_range or ys must be passed.")

    if ys is not None:
        ys = ys.flatten()
        ys = ys[~torch.isnan(ys)]
        if len(ys) <= num_outputs:
            raise ValueError(
                f"Number of ys: {len(ys)} must be larger than "
                f"num_outputs: {num_outputs}"
            )
        if len(ys) % num_outputs:
            ys = ys[: -(len(ys) % num_outputs)]
        ys_per_bucket = len(ys) // num_outputs
        if full_range is None:
            full_range = (ys.min(), ys.max())
        else:
            if full_range[0] > ys.min() or full_range[1] < ys.max():
                raise ValueError("full_range must cover all target values.")
            full_range = torch.tensor(full_range)  # type: ignore

        ys_sorted, _ = ys.sort(0)  # type: ignore
        bucket_limits = (
            ys_sorted[ys_per_bucket - 1 :: ys_per_bucket][:-1]
            + ys_sorted[ys_per_bucket::ys_per_bucket]
        ) / 2
        bucket_limits = torch.cat(
            [full_range[0].unsqueeze(0), bucket_limits, full_range[1].unsqueeze(0)],  # type: ignore
            0,
        )
        if widen_bucket_limits_factor is not None:
            bucket_limits = bucket_limits * widen_bucket_limits_factor

    else:
        class_width = (full_range[1] - full_range[0]) / num_outputs  # type: ignore
        bucket_limits = torch.cat(
            [
                full_range[0] + torch.arange(num_outputs).float() * class_width,  # type: ignore
                torch.tensor(full_range[1]).unsqueeze(0),  # type: ignore
            ],
            0,
        )

    if len(bucket_limits) - 1 != num_outputs:
        raise RuntimeError(
            f"len(bucket_limits) - 1 == {len(bucket_limits) - 1}"
            f" != {num_outputs} == num_outputs"
        )

    if not widen_bucket_limits_factor or widen_bucket_limits_factor == 1.0:
        if full_range[0] != bucket_limits[0]:  # type: ignore
            raise RuntimeError(f"{full_range[0]} != {bucket_limits[0]}")  # type: ignore
        if full_range[-1] != bucket_limits[-1]:  # type: ignore
            raise RuntimeError(f"{full_range[-1]} != {bucket_limits[-1]}")  # type: ignore

    return bucket_limits


def get_custom_bar_dist(borders: torch.Tensor, criterion: nn.Module) -> nn.Module:
    # Tested that a bar_dist with borders 0.54 (-> softplus 1.0) yields the same
    # bar distribution as the passed one.
    borders_ = torch.nn.functional.softplus(borders) + 0.001
    borders_ = torch.cumsum(
        torch.cat([criterion.borders[0:1], criterion.bucket_widths]) * borders_,
        0,
    )
    return criterion.__class__(
        borders=borders_,
        ignore_nan_targets=getattr(criterion, "ignore_nan_targets", True),
        reduction=getattr(criterion, "reduction", "none"),
    )
