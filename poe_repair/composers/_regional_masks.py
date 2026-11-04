"""Two boxes to two soft masks on the latent grid, summing to 1 at every cell.

The regional composer gives each expert its own cells instead of letting both constrain every
cell. That needs a weight per expert per cell, and the weights have to sum to 1 so no cell is
pushed twice as hard as its neighbour. These are the three steps:

1. **Indicator.** 1 inside the box's cells, 0 outside.
2. **Blur.** A Gaussian, so the boundary is a ramp rather than a cliff. A hard edge in the noise
   prediction shows up as a seam in the render. ``EDGE_CELLS`` is the 10%-to-90% width of that
   ramp in cells; a Gaussian blur of a step edge has a 10-90 width of about 2.563 sigma, which
   is where ``sigma_for`` comes from.
3. **Normalise.** Divide by the sum, so the pair sums to 1 everywhere.

**The overlap has two readings and they are not equivalent.** Where the boxes cross, step 3 on
its own splits that region evenly, because both indicators are 1 there. The alternative is to
give the overlap to the expert whose box is smaller, on the grounds that the smaller animal is
the one at risk of being swallowed. Which to use is a question the renders answer, so both are
built and both are rendered before the sweep.

**Outside both boxes** the blurred indicators are near zero and the normalisation is undefined,
so those cells are split evenly. They are background: neither expert owns them, and both act at
half strength there.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

LATENT = 128
EDGE_CELLS = 4.0          # the 10%-to-90% width of the blurred boundary, in latent cells
STEP_10_90_PER_SIGMA = 2.563   # a Gaussian blur of a step edge: its 10-90 width in sigmas
EMPTY = 1e-6              # below this a cell is outside both boxes and the split is even

READINGS = ("even", "smaller-wins")


def sigma_for(edge_cells: float = EDGE_CELLS) -> float:
    """The Gaussian sigma whose blurred step edge ramps over ``edge_cells`` cells."""
    return edge_cells / STEP_10_90_PER_SIGMA


@dataclass(frozen=True)
class MaskPair:
    """Two masks over the latent grid, and what they were built from."""

    a: np.ndarray          # the first expert's weight per cell, in [0, 1]
    b: np.ndarray          # the second expert's, with a + b == 1 everywhere
    reading: str
    box_a_cells: tuple[int, int, int, int]
    box_b_cells: tuple[int, int, int, int]
    edge_cells: float

    @property
    def max_sum_deviation(self) -> float:
        """How far the pair is from summing to 1, at its worst cell."""
        return float(np.abs(self.a + self.b - 1.0).max())

    @property
    def share_a(self) -> float:
        """The first expert's share of the whole grid, as a fraction of the cells."""
        return float(self.a.mean())


def indicator(box, grid: int = LATENT) -> np.ndarray:
    """1 inside the box's cells, 0 outside. ``box`` is ``(x0, y0, x1, y1)`` in cells."""
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    out = np.zeros((grid, grid), dtype=np.float64)
    out[max(0, y0):min(grid, y1), max(0, x0):min(grid, x1)] = 1.0
    return out


def masks_from_boxes(box_a, box_b, *, reading: str = "even", grid: int = LATENT,
                     edge_cells: float = EDGE_CELLS) -> MaskPair:
    """Build the two masks for one seed, in one of the two overlap readings.

    ``reading="even"`` splits the overlap between the two experts. ``reading="smaller-wins"``
    removes the smaller box from the larger one before blurring, so the overlap goes to the
    smaller box's expert.
    """
    if reading not in READINGS:
        raise ValueError(f"reading must be one of {READINGS}, not {reading!r}")
    ind_a, ind_b = indicator(box_a, grid), indicator(box_b, grid)

    if reading == "smaller-wins":
        if ind_a.sum() <= ind_b.sum():
            ind_b = ind_b * (1.0 - ind_a)
        else:
            ind_a = ind_a * (1.0 - ind_b)

    sigma = sigma_for(edge_cells)
    soft_a = gaussian_filter(ind_a, sigma=sigma, mode="constant", cval=0.0)
    soft_b = gaussian_filter(ind_b, sigma=sigma, mode="constant", cval=0.0)

    total = soft_a + soft_b
    empty = total < EMPTY
    a = np.where(empty, 0.5, soft_a / np.where(empty, 1.0, total))
    b = 1.0 - a
    return MaskPair(
        a=a, b=b, reading=reading,
        box_a_cells=tuple(int(round(v)) for v in box_a),
        box_b_cells=tuple(int(round(v)) for v in box_b),
        edge_cells=edge_cells,
    )
