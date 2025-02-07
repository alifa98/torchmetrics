# Copyright The Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from collections.abc import Sequence
from typing import Any, Optional, Union

import torch
from torch import Tensor
from typing_extensions import Literal

from torchmetrics.functional.regression.js_divergence import _jsd_compute, _jsd_update
from torchmetrics.metric import Metric
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["JSDivergence.plot"]


class JSDivergence(Metric):
    r"""Compute the `JS divergence`_.

    The Jensen-Shannon divergence is defined as a symmetric and finite measure of similarity between two probability
    distributions. It is defined in terms of the Kullback-Leibler divergence as follows:

    .. math::
        JS(P||Q) = \frac{1}{2} D_{KL}\Big(P \,\big\|\, \frac{1}{2}(P+Q)\Big)
                 + \frac{1}{2} D_{KL}\Big(Q \,\big\|\, \frac{1}{2}(P+Q)\Big)
        \quad \text{with} \quad M = \frac{1}{2}(P+Q).

    As input to ``forward`` and ``update``, the metric accepts the following inputs:

    - ``p`` (:class:`~torch.Tensor`): a distribution with shape ``(N, d)``
    - ``q`` (:class:`~torch.Tensor`): another distribution with shape ``(N, d)``

    As output of ``forward`` and ``compute`` the metric returns the following output:

    - ``js_divergence`` (:class:`~torch.Tensor`): A tensor with the Jensen-Shannon divergence

    Args:
        log_prob: bool indicating if the input is log-probabilities or probabilities. If provided as probabilities,
            they will be normalized so that each distribution sums to 1.
        reduction:
            Determines how to reduce over the ``N``/batch dimension:

            - ``'mean'`` [default]: Averages the scores across samples.
            - ``'sum'``: Sums the scores across samples.
            - ``'none'`` or ``None``: Returns the score per sample.
        kwargs: Additional keyword arguments (see :ref:`Metric kwargs` for more info).

    Raises:
        TypeError:
            If ``log_prob`` is not a ``bool``.
        ValueError:
            If ``reduction`` is not one of ``'mean'``, ``'sum'``, ``'none'`` or ``None``.

    .. attention::
        Half precision is only supported on GPU for this metric.

    Example:
        >>> from torch import tensor
        >>> from torchmetrics.regression import JSDivergence
        >>> p = tensor([[0.36, 0.48, 0.16]])
        >>> q = tensor([[1/3, 1/3, 1/3]])
        >>> js_divergence = JSDivergence()
        >>> js_divergence(p, q)
        tensor(0.0571)

    """

    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0

    total: Tensor

    def __init__(
        self,
        log_prob: bool = False,
        reduction: Literal["mean", "sum", "none", None] = "mean",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if not isinstance(log_prob, bool):
            raise TypeError(f"Expected argument `log_prob` to be bool but got {log_prob}")
        self.log_prob = log_prob

        allowed_reduction = ["mean", "sum", "none", None]
        if reduction not in allowed_reduction:
            raise ValueError(f"Expected argument `reduction` to be one of {allowed_reduction} but got {reduction}")
        self.reduction = reduction

        if self.reduction in ["mean", "sum"]:
            self.add_state("measures", torch.tensor(0.0), dist_reduce_fx="sum")
        else:
            self.add_state("measures", [], dist_reduce_fx="cat")
        self.add_state("total", torch.tensor(0), dist_reduce_fx="sum")

    def update(self, p: Tensor, q: Tensor) -> None:
        """Update metric states with predictions and targets."""
        measures, total = _jsd_update(p, q, self.log_prob)
        if self.reduction is None or self.reduction == "none":
            self.measures.append(measures)
        else:
            self.measures += measures.sum()
            self.total += total

    def compute(self) -> Tensor:
        """Compute metric."""
        measures: Tensor = dim_zero_cat(self.measures) if self.reduction in ["none", None] else self.measures
        return _jsd_compute(measures, self.total, self.reduction)

    def plot(
        self, val: Optional[Union[Tensor, Sequence[Tensor]]] = None, ax: Optional[_AX_TYPE] = None
    ) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric.

        Args:
            val: Either a single result from calling `metric.forward` or `metric.compute` or a list of these results.
                If no value is provided, the method will automatically call `metric.compute` and plot that result.
            ax: A matplotlib axis object. If provided, the plot will be added to that axis.

        Returns:
            Figure and Axes object

        Raises:
            ModuleNotFoundError:
                If `matplotlib` is not installed.

        .. plot::
            :scale: 75

            >>> from torch import randn
            >>> # Example plotting a single value
            >>> from torchmetrics.regression import JSDivergence
            >>> metric = JSDivergence()
            >>> metric.update(randn(10, 3).softmax(dim=-1), randn(10, 3).softmax(dim=-1))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> from torch import randn
            >>> # Example plotting multiple values
            >>> from torchmetrics.regression import JSDivergence
            >>> metric = JSDivergence()
            >>> values = []
            >>> for _ in range(10):
            ...     values.append(metric(randn(10, 3).softmax(dim=-1), randn(10, 3).softmax(dim=-1)))
            >>> fig, ax = metric.plot(values)

        """
        return self._plot(val, ax)
