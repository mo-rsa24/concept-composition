# Attend-and-Excite evaluation prompts

`a.e_prompts.json` holds the prompt sets Chefer et al. (2023) used for their quantitative evaluation, and that the CO3 paper (arXiv 2509.25940) scores in its Table 1: `animals` (66 prompts, "a dog and a frog"), `animals_objects` (144, "a horse with a glasses") and `objects` (66, "a pink crown and a purple bow").

Downloaded 2026-11-10 from https://github.com/AttendAndExcite/Attend-and-Excite/files/11336216/a.e_prompts.txt, the attachment the repository's README links; the repository itself does not contain it. Saved byte for byte, renamed to `.json` because its content is JSON.

Used for slot R3, the CO3-protocol table (`report/paper-review/layout-sheet.md`). None of the 276 prompts names one of the 48 training pairs of `pair_pool_v58.yaml`; the animals set contains "a cat and a dog", a showcase pair under the protocol, which R3 reports apart.
