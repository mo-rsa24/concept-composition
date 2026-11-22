# Repairing the classifier push in the SDXL product sampler

The walk replaced the product sampler's rescaled classifier push with the unrescaled form and capped it at 0.1 of the product prediction's norm. The walk is closed and passed its check. At strength 0, the repaired sampler and plain product render cat × dog seed 9 with 0 pixels differing (md5 `c291b259647d30a450e87b234423fe99`).

**What it fed:** the [sampler repair entry on the classifier-guided PoE board](../../../report/classifier-guided-poe/board.md#outcomes-per-row).

**Verdict:** DRAIN THEN ARCHIVE `proposed`. The walk is closed and its result is on that board.
