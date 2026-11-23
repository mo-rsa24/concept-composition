# How each picture is called in the blind count (step M1, Paper 2)

This page fixes the calls before any picture from the comparison is counted. The [M1 design](design.md) names the bar these calls feed.

## What the caller sees

Each picture has a neutral 12-character id and its prompt, such as "a puma and a condor". The first subject is A and the second is B. The
caller opens one picture at a time at its full 1024 by 1024 size and is never told which method drew it.

## The rung

Pick the highest rung the picture fully meets.

| Rung | What the picture holds |
|---|---|
| `one` | One subject, or neither subject |
| `hybrid` | One body that mixes features of A and B |
| `two_bodies` | Two or more separate bodies or heads that fail `two_clear`: one borrows the other's face or parts, both are the same subject, one is the wrong subject, or there is more than one A or more than one B |
| `two_clear` | Exactly one clear A and exactly one clear B as separate bodies, with no borrowed features |

A body hidden so far that its subject cannot be named does not count as clear. When two rungs both seem possible, take the lower one.

## The two marks

**`damaged` is 1** when the picture is not an intact photograph: melted or broken anatomy, garbled faces, stray limbs, or a drawing,
cartoon, poster or collage.

**`faithful` is 1** only when the rung is `two_clear` and `damaged` is 0. This is the count the bar uses.

## What gets written

One CSV row per picture, with the columns `id`, `rung`, `faithful`, `damaged` and `note`. The note is a few words saying what is in the
picture. It names no method and compares with no other picture.
