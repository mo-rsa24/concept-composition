# Replies to Reviewers k4nd and UHz6

Submission 56173. One section per reviewer point: the reviewer's words quoted, then the reply.
Equation, section and figure numbers refer to the revised paper
(`paper/iclr-56173/iclr2027_conference.tex`). k4nd's review is pages 4 and 5 of `Reviews.pdf`;
UHz6's is page 3. A reply marked pending has not been agreed yet.

## Reviewer k4nd

### W1: the residual does not establish a distinct representation of plurality

> "The proposed residual does not establish a distinct representation of plurality. … The paper
> does not establish that it specifically measures whether the named concepts appear as distinct
> objects."

Pending.

### W2: methodological novelty

> "The methodological novelty is insufficiently established."

Pending.

### W3: no systematic quantitative evaluation of the adapter

> "The learned adapter lacks systematic quantitative evaluation."

Pending.

### W4: method details poorly presented and sometimes inconsistent

**The training loss is not stated in Section 4**

> "Section 4 describes target matching and provides the teacher prediction and sampling rule, but
> it does not explicitly formulate the training loss. The squared-error objective appears in the
> training diagram in Appendix A, Figure 8."

Section 4 now states the training loss as Equation 12: the squared error between the adapted
composition and the joint-prompt prediction, over the cached states of every training pair and
seed at the first ten of the 50 sampling steps, with each step's term weighted by
$(1-\bar\alpha_t)/\bar\alpha_t$ divided by its mean over the 50-step grid. The training diagram
is removed until its notation matches Equation 12.

**Evaluations described as returning the correction**

> "The main text describes individual adapted network evaluations as returning the correction,
> whereas Appendix B defines them as adapted denoising predictions whose composition induces the
> correction. Section 4 refers to 'Figure 10,' although the actual training diagram is Figure 8."

Section 4 now says each adapted evaluation returns a noise prediction, as Appendix B derives. At
inference the sampler uses the adapted composition directly (Equation 15): three predictions per
step, and never the joint prompt. The correction $\hat r_\psi$ is the implicit difference from the
unadapted composition (Equation 13), and Equation 13 at $\lambda = 1$ is the adapted composition
itself. The reference that printed as "Figure 10" is removed.

**The layers the adapter modifies, and its size**

The method text, abstract and parameter count now describe the adapter as trained: 35.8 M
parameters on the query, key and value projections of both the self-attention and cross-attention
layers. The submission said cross-attention only and 19.8 M.

**The training schedule**

Pending.

**Repetitive claims about learning an abstract concept**

> "The repetitive claims about learning an abstract concept occupy space that could instead
> clarify the objective, training data, evaluation protocol, and limitations."

We removed the repeated claims about learning an abstract concept from Sections 3 and 4 and the
Discussion. The introduction now states what the correction is and what it restores (see W1).

### W5: the qualitative baseline comparison and its starting noise

> "The Figure 5 caption states that CO3 uses different starting noise, while Section 4.1 states
> that all examples use the same initial seeds."

Pending.

### Q1: evidence that the residual measures plurality

> "What evidence establishes that the residual in Equation (9) specifically measures plurality
> rather than the broader difference between joint and separate conditioning?"

Pending, with W1.

### Q2: aggregate results over held-out concepts and seeds

> "Can the authors provide aggregate results for the learned adapter and baselines over documented
> held-out concepts and multiple seeds, including identity errors, additional objects, and
> style-dependent failures?"

Pending, with W3. The numbers come from route steps 5, 9 and 12.

### Q3: pairs retained when only CO3 succeeds

> "Training pairs are retained when either joint prompting or CO3 succeeds, but supervision comes
> from the frozen joint-prompt prediction. How are pairs handled when only CO3 succeeds?"

Pending.

### Q4: the objective in the main text, and the inconsistencies

> "Can the authors state the optimization objective explicitly in the main text and resolve the
> inconsistencies concerning network outputs, figure references, and baseline starting noise?"

The objective is Equation 12 in Section 4. The network outputs and the figure reference are
resolved as described under W4. Baseline starting noise: pending, with W5.

## Reviewer UHz6

### Presentation issues

> "There are presentation issues: Sec. 4 refers to a Figure 10 that does not exist, and several
> sentences are broken (e.g., "inside create hybrid objects" in Sec. 5)."

The reference that printed as "Figure 10" in Section 4 is removed. "inside create": pending.

### Q1: the 48 training pairs, the held-out pairs, and whether cat and dog is trained on

> "Could the authors list the 48 training pairs and the held-out pairs, and clarify whether
> cat/dog is in the training set?"

Pending.

### Q2: success rate against PoE and CO3 over all held-out pairs and seeds

> "What is the success rate of the adapter compared with PoE and CO3 over all held-out pairs and
> multiple seeds?"

Pending. The numbers come from route steps 5, 9 and 12.

### Q3: restricting the adapter to the early steps

> "Given the finding in Sec. 3, does restricting the adapter to the early steps change the failure
> modes in Fig. 7?"

Pending (slot R9).

### Q4: pairs that do not compete for the same region

> "Does the adapter degrade PoE on pairs that do not compete for the same region, such as
> foreground/background pairs like lighthouse and sea?"

Pending (slot R3, the not-competing group of 21 pairs).
