from typing import Union

import torch
from torch import Tensor
from typing_extensions import Literal

from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.compute import _safe_xlogy


def _jsd_update(p: Tensor, q: Tensor, log_prob: bool) -> tuple[Tensor, int]:
    """Update and return JS divergence scores for each observation and the total number of observations.

    Args:
        p: First probability distribution with shape ``[N, d]``
        q: Second probability distribution with shape ``[N, d]``
        log_prob: Boolean indicating if the inputs are log-probabilities or probabilities.
            If given as probabilities, they will be normalized so that each distribution sums to 1.
    """
    _check_same_shape(p, q)
    if p.ndim != 2 or q.ndim != 2:
        raise ValueError(
            f"Expected both p and q distributions to be 2D but got {p.ndim} and {q.ndim} respectively"
        )

    total = p.shape[0]
    if log_prob:
        p_prob = p.exp()
        q_prob = q.exp()
        m = torch.log(0.5 * (p_prob + q_prob))
        measure1 = torch.sum(p_prob * (p - m), dim=-1)
        measure2 = torch.sum(q_prob * (q - m), dim=-1)
        measures = 0.5 * (measure1 + measure2)
    else:
        p = p / p.sum(dim=-1, keepdim=True)
        q = q / q.sum(dim=-1, keepdim=True)
        m = 0.5 * (p + q)
        measure1 = _safe_xlogy(p, p / m).sum(dim=-1)
        measure2 = _safe_xlogy(q, q / m).sum(dim=-1)
        measures = 0.5 * (measure1 + measure2)

    return measures, total


def _jsd_compute(
    measures: Tensor,
    total: Union[int, Tensor],
    reduction: Literal["mean", "sum", "none", None] = "mean",
) -> Tensor:
    """Compute the JS divergence based on the type of reduction.

    Args:
        measures: Tensor of JS divergence scores for each observation.
        total: Number of observations.
        reduction:
            Determines how to reduce over the batch (N) dimension:

            - ``'mean'`` [default]: Averages scores across samples.
            - ``'sum'``: Sums scores across samples.
            - ``'none'`` or ``None``: Returns score per sample.

    """
    if reduction == "sum":
        return measures.sum()
    if reduction == "mean":
        return measures.sum() / total
    if reduction is None or reduction == "none":
        return measures
    return measures / total


def js_divergence(
    p: Tensor,
    q: Tensor,
    log_prob: bool = False,
    reduction: Literal["mean", "sum", "none", None] = "mean",
) -> Tensor:
    r"""Compute the `Jensen-Shannon divergence`_.

    The Jensen-Shannon divergence is a symmetric and smoothed version of the KL divergence and is defined as:

    .. math::
        D_{JS}(P\|Q) = \frac{1}{2}\, D_{KL}(P\|M) + \frac{1}{2}\, D_{KL}(Q\|M)
        \quad \text{with} \quad M = \frac{1}{2}(P+Q)

    where :math:`P` and :math:`Q` are probability distributions. The JS divergence is bounded
    between 0 and \(\log(2)\) (using the natural logarithm).

    Args:
        p: First probability distribution with shape ``[N, d]``
        q: Second probability distribution with shape ``[N, d]``
        log_prob: Boolean indicating if the inputs are log-probabilities or probabilities.
            If given as probabilities, they will be normalized so that each distribution sums to 1.
        reduction:
            Determines how to reduce over the batch (N) dimension:

            - ``'mean'`` [default]: Averages scores across samples.
            - ``'sum'``: Sums scores across samples.
            - ``'none'`` or ``None``: Returns score per sample.

    Example:
        >>> import torch
        >>> p = torch.tensor([[0.36, 0.48, 0.16]])
        >>> q = torch.tensor([[1/3, 1/3, 1/3]])
        >>> js_divergence(p, q)
        tensor(0.0197)

    """
    measures, total = _jsd_update(p, q, log_prob)
    return _jsd_compute(measures, total, reduction)
