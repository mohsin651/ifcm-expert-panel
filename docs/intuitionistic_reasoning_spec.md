# Issue 2 — proposed signed intuitionistic reasoning specification

Status: implemented as pure NumPy functions in
`ifcm/intuitionistic_reasoning.py`, with synthetic tests in
`tests/test_intuitionistic_reasoning.py`. No A/B1/B2 dataset run, calibration,
GPU job, test-label inspection, or publication result was produced here. This
is a **proposed research operator**, not an established clinical rule. Its
signed propagation and conflict projection are novel adaptations; they must
be described as such in any paper and checked by the research lead before a
strict outer-test comparison is interpreted.

## Scientific context and source of the operations

An Atanassov intuitionistic fuzzy value (IFS) has membership `mu`,
non-membership `nu`, and residual hesitation `pi=1-mu-nu`, with each component
in `[0,1]` and `mu+nu<=1` ([Atanassov, 1986](https://doi.org/10.1016/S0165-0114(86)80034-3)).
It corresponds to a membership interval `[L,U]=[mu,1-nu]`.
Product t-norm and its dual probabilistic-sum t-conorm
`S(a,b)=a+b-ab=1-(1-a)(1-b)` are established ingredients of IFS
aggregation. The interval-valued correspondence and limitations of several
standard averaging operators are discussed by
[Beliakov et al., 2011](https://doi.org/10.1016/j.ins.2010.11.024).
The **choice to split a signed clinical edge into positive and inhibitory
evidence and then project their conflict to a diagnosis IFS below is ours**;
those sources do not validate it for medical diagnosis.

## Inputs and IFS encoding

All arrays must be finite and explicitly aligned by ordered concept and
diagnosis names before these pure functions are called. A missing edge is an
error; a present edge with weight zero is neutral.

| Quantity | Shape | Rule |
| --- | --- | --- |
| Finding mean `x` | `(N,C)` | Same seed mean used by Model A; probability-like score in `[0,1]`. |
| Finding draws | `(N,C,S)` | Five CNN seeds for strict LIDC, 30 probes for existing Derm/frozen pathways; not independent posterior samples. |
| Finding IFS | three `(N,C)` arrays | From `[xL,xU]`: `mu_x=xL`, `nu_x=1-xU`, `pi_x=xU-xL`. |
| B1 edges | `(C,D)` | Full-precision panel mean `W`, signed in `[-1,1]`, with `wL=wU=W`. |
| B2 edges | two `(C,D)` arrays | Same mean, with pre-specified `wL<=W<=wU` built from panel disagreement/unsure. |
| Diagnosis result | three `(N,D)` arrays | `mu_d`, `nu_d`, `pi_d`, plus scalar score and evidence diagnostics. |

For CNN seeds or other aligned draws, the fixed, label-free heuristic is
`qL=empirical 10th percentile`, `qU=empirical 90th percentile`,
`xL=min(qL,mean(draws))`, `xU=max(qU,mean(draws))`. The mean-inclusion envelope
guards against five-seed skew. Provided `finding_mean` must equal the seed
mean within `1e-6` floating-point tolerance, avoiding silent changes to
Model A's concept input. If draws are absent,
use the zero-hesitancy point value `(x,1-x,0)`. The 10th/90th seed envelope is
**not** a confidence, credible, or calibrated prediction interval; five seeds
cannot justify those claims. This fixed rule was chosen without looking at
held-out diagnostic outcomes. Any alternative width or quantiles require a
new, separately named protocol selected within the fit/validation group only.

For B2, let `w_m` and `u_m` be aligned per-member signed edge weights and
unsure probabilities for `M>=2` LLM-panel members. The pre-specified heuristic
is `W=mean_m(w_m)`, `h_raw=mean_m(u_m)+sample_SD_m(w_m)`, and
`wL=clip(W-h_raw,-1,1)`, `wU=clip(W+h_raw,-1,1)`.
Record `h_raw`, `W-wL`, and `wU-W` separately: clipping near ±1 makes the
effective left/right widths asymmetric. This is the existing panel-style
uncertainty proxy, **not** a learned likelihood, calibrated confidence bound,
or credible interval. Issue 4 still has to reconstruct exact full-precision
member matrices from the saved votes and verify their order; this module
does not read vote files or regenerate LLM answers.

## Signed evidence transfer and aggregation

For sample `n`, finding `c`, diagnosis `d`, let `xL,xU` and `wL,wU` be the
interval endpoints. The positive and negative evidence ranges are

```text
pL = xL * max(wL, 0)       pU = xU * max(wU, 0)
rL = xL * max(-wU, 0)      rU = xU * max(-wL, 0)
```

Here `p` supports the diagnosis and `r` opposes it. A negative edge therefore
never masquerades as positive support. A sign-crossing edge may offer *possible*
support and *possible* opposition, but its guaranteed lower evidence on both
sides is zero. An absent finding `xL=xU=0` contributes nothing; an unknown
finding `[0,1]` may contribute possible but not guaranteed evidence. A zero
edge contributes neither, while missing/NaN edges are rejected.

Aggregate each side and each endpoint over findings using the established
probabilistic-sum t-conorm, `OR(q_1,...,q_C)=1-prod_c(1-q_c)`:

```text
PL = OR_c(pL)   PU = OR_c(pU)
RL = OR_c(rL)   RU = OR_c(rU)
mu_d = PL * (1-RU)
nu_d = RL * (1-PU)
pi_d = 1-mu_d-nu_d
s_d = mu_d-nu_d
```

The `mu/nu` conflict projection is our proposed signed extension. It is
deliberately conservative: guaranteed positive evidence is discounted by
*possible* opposition, and vice versa. Since `PL<=PU`, `RL<=RU`, and all are
in `[0,1]`, `mu+nu` is no greater than
`PU(1-RU)+RU(1-PU)<=1`; hence `pi>=0`. The score `s=mu-nu` lies in `[-1,1]`.
We compute `OR` using `-expm1(sum(log1p(-q)))` for numerical stability and
exactly handle a contribution of one.

**Interpretation limit:** diagnosis `pi` combines missing support, opposing
evidence, and interval uncertainty. It is not solely CNN uncertainty, solely
LLM disagreement, a probability of diagnostic error, or the old perceptual/
knowledge decision-flip frequency. Even point-valued findings and edges can
yield nonzero `pi`. Save `PL,PU,RL,RU` to distinguish the contributors.

## Worked examples and limits

1. Point findings `x=(0.8,0.5)` with edges `(0.6,-0.4)` give `P=0.48`,
   `R=0.20`, so `(mu,nu,pi)=(0.384,0.104,0.512)` and `s=0.28`. The negative
   edge lowers support and raises opposition, rather than being treated as a
   second positive association.
2. With both findings in `[0.6,0.9]`, B1 point edges `(0.6,-0.4)` give
   `PL=0.36, PU=0.54, RL=0.24, RU=0.36`, hence
   `(mu,nu,pi)=(0.2304,0.1104,0.6592)`. B2 intervals
   `w1=[0.5,0.7]`, `w2=[-0.5,-0.3]` give
   `(PL,PU,RL,RU)=(0.30,0.63,0.18,0.45)` and
   `(mu,nu,pi)=(0.165,0.0666,0.7684)`. The wider edge bounds increase
   residual hesitation in this case; this is not a universal improvement in
   diagnostic accuracy.
3. Two certain, opposed unit edges with findings equal to one produce
   `P=R=1` and `(mu,nu,pi)=(0,0,1)`: maximal conflict. A neutral edge or an
   absent finding also gives `(0,0,1)`, but from *lack of evidence*, not
   conflict; the saved evidence diagnostics distinguish the cases.
4. With `wL=wU=W`, B2 calls exactly the B1 computation. Thus B2 reduces to
   B1 when edge uncertainty is zero, for uncertain or point-valued findings.
   With point-valued `x`, a single edge has `s=x*w`; for several weak edges,
   `s` approaches `sum_c x_c*w_cd` to first order, but generally differs due
   to nonlinear evidence aggregation and conflict. It must not be presented as
   algebraically identical to Model A.

Increasing a positive point edge or positive finding cannot decrease the
score when other evidence is fixed; increasing negative-edge magnitude cannot
increase it. Widening a nested finding or edge interval makes both `mu` and
`nu` non-increasing and `pi` non-decreasing, but **the score need not be
monotone in interval width** because both support and opposition can shrink.
The t-conorm counts correlated findings separately; duplicate concepts can
overstate evidence. There is no claim of probabilistic independence.

## Calibration and fair comparison with Model A

Use the same sample IDs, concepts, full-precision panel, split, and prediction
views for A, B1 and B2. Model A remains `z_A=X_mean@W_mean`. B1/B2 feed their
single scalar `s_d` to **one slope plus one intercept per diagnosis**, matching
Model A's logistic-calibration parameter count. Fit calibrators only on the
approved fit/inner-OOF data. Derm retains four independent sigmoids and their
existing argmax rule. LIDC retains independently calibrated MAL/BEN outputs,
then normalizes `p_MAL/(p_MAL+p_BEN)` and uses its fixed threshold. Any
selection of interval mapping/operator/calibration options must occur within
the recorded fit-patient selection subset; after freezing, final calibrators
may refit on the approved fit pool. The strict outer-test diagnosis labels
must not influence any of these choices. This issue does **not** fit a
calibrator; Issues 3–5 must implement and verify that boundary.

An empirical B1/B2 improvement is **not assumed**. The operator may merely
create a nonlinear transformation of Model A's score, and a two-parameter
sigmoid could already absorb some monotone changes. Key ablations must test
point-valued findings, B2 zero edge width, fixed calibration capacity, and
paired A/B1/B2 predictions. Any purported gain must be evaluated on aligned
cases with patient-level uncertainty, not chosen by inspecting outer-test
performance. No significance claim is justified by this specification alone.

## Alternatives rejected for this pre-registered candidate

- Applying unsigned IFWA/t-norms directly to `w∈[-1,1]` is invalid: their
  domain is `[0,1]` and the sign/inhibitory meaning is lost. A shift such as
  `(w+1)/2` makes a zero edge appear as positive evidence.
- Ordinary interval arithmetic on `sum(x*w)` followed by a sigmoid yields
  valid endpoint bounds, but the diagnosis score is essentially Model A with
  interval decoration; it does not explicitly aggregate positive and
  inhibitory IFS evidence. It remains a useful simpler comparator, not B1.
- Standard IFWA over finding triples does not itself explain signed causal
  edges or clinical conflict, and some established averaging variants have
  undesirable ordinary-fuzzy limiting behavior
  ([Beliakov et al., 2011](https://doi.org/10.1016/j.ins.2010.11.024)).
- Directly setting `mu=positive_OR`, `nu=negative_OR` fails the IFS constraint
  when both sides are strong (`mu+nu>1`). The conflict projection above is
  chosen to preserve validity and expose disagreement as `pi`.

## Pure API and handoff

`point_ifs(x)` and `from_interval(lo,hi)` return validated `IFS` values.
`finding_ifs_from_draws(draws,mean)` returns the fixed empirical envelope.
`panel_edge_bounds(member_edges,member_unsure)` returns full-precision mean,
clipped bounds, raw half-width, and effective asymmetric widths.
`reason_b1(findings,edge_mean)` and
`reason_b2(findings,edge_lower,edge_upper)` return `ReasoningResult` with
diagnosis `IFS`, scalar score, and positive/negative evidence endpoints.
These functions are CPU-only, deterministic, do no I/O, accept no labels, and
reject misaligned shapes, non-finite values, and out-of-range probabilities.

Run the focused synthetic suite with:

```bash
../.venv-repro/bin/python -m unittest tests.test_intuitionistic_reasoning
```

Issues 3 and 4 must still implement dataset adapters, panel reconstruction,
fit-only calibration, immutable manifests, and CLI-driven prediction. Do not
describe these pure functions as a completed A/B1/B2 experiment or put their
toy values in a clinical results table.
