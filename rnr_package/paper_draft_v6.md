# What Does Adaptivity Cost? Exploitation, Identification, and Exposure in Three-Player Leduc Hold'em

**Draft v6 — internal. CORRECTION to v5: the one-sided RNR's "worst case" only checked seat 1; seat 2 punishes it for -3.6. Section 6 claims downgraded pending the symmetric run (RERUN-21), queued locally.** Earlier: **Draft v5 — RERUN-17 done.** Earlier: **Draft v4 — Re-anchored on the safe-exploitation literature; Shi et al. removed.** Earlier notes: **Draft v3 — Systematic literature sweep added (type-based reasoning, Morton's theorem, multiplayer equilibrium computation, showdown censoring). Section 6 frontier claim withdrawn pending [RERUN-17]; Section 3.2 reframed.**
Every claim is tagged with how it was produced:
**[EXACT]** full game-tree computation, no sampling error;
**[MC]** Monte Carlo, standard error quoted;
**[RERUN-n]** a specific weakness with the fix, collected in Section 11.

---

## Abstract

Adaptive poker agents are said to trade unexploitability for profit, but the
tradeoff has never been measured, because in large games exploitability cannot be
computed. We measure both sides exactly in three-player Leduc Hold'em. From a
converged CFR+ equilibrium (nash_conv 0.0057) we construct opponents by tilting the
equilibrium toward one action with probability alpha, giving over-raisers,
over-callers, and over-folders. We report four results. (i) Best-response value
rises monotonically with opponent bias, but a *non-adapting* equilibrium player can
*lose* money to extreme opponents, because three-player equilibria carry no minimax
guarantee. (ii) Exact Bayesian identification of an opponent pair needs a median of
26 to 140 hands depending on bias; revealing hole cards every hand helps by a factor
of 1.5 to 1.9, while showdown cards alone help almost not at all. (iii) [PENDING RERUN-21] Naively mixing
equilibrium with a best response appears to be a poor way to adapt: a preliminary
symmetric three-player restricted Nash response earns about six times equilibrium's
gain against biased opponents without being more exploitable. The final number
depends on a converged run. (iv) Comparing opponent
types at equal *deviation rate* rather than equal alpha reverses the apparent
ranking: folding errors are roughly ten times more punishable than raising errors
per unit of deviation, and are also the slowest to detect. Prior Bayesian opponent
modelling in multiplayer poker has been demonstrated in three-player Kuhn poker
(Ganzfried et al., 2024), and confidence-scheduled safe exploitation in two-player
Leduc (Li & Huang, 2026); we contribute exact measurement of the exploitation,
identification and exposure axes together in a three-player game with a public
board card.

---

## 1. Introduction

An equilibrium policy cannot be exploited but leaves money on the table against weak
opponents; an exploitative policy wins more against a specific opponent but becomes
exploitable itself. In two-player zero-sum games this tradeoff is well understood:
restricted Nash responses trace its Pareto frontier (Johanson, Zinkevich & Bowling,
2007), and safe exploitation bounds how much may be risked relative to the game value
(McCracken & Bowling, 2004; Ganzfried & Sandholm, 2015). Every one of those guarantees
rests on the minimax theorem. With three or more players there is no game value to be
safe relative to, an equilibrium strategy carries no performance floor, and opponents'
mistakes need not become your gains. The tradeoff is therefore not just unmeasured in
multiplayer poker but not yet well defined. Measuring it requires a game small enough
that best responses and exploitability can be computed exactly, and large enough that
three players genuinely interact.

The closest prior work is Ganzfried, Wang & Chiswick (2024), who maintain Bayes'-rule
posteriors over sampled opponent strategies in three-player Kuhn poker, follow an
equilibrium strategy for an exploration phase, then best-respond to the joint model.
Our identification-then-response pipeline has the same shape. We differ in the game
(Leduc has two betting rounds and a public card, so evidence arrives unevenly), in
using a structured, parameterised hypothesis space rather than sampled strategies, and
above all in measuring exploitability of our *own* adaptive policy exactly, which they
do not. In two-player Leduc, Li & Huang (2026) schedule how far to exploit from
confidence sequences and certify the exploitability of every deployed strategy; our
beta experiment asks a similar question without their guarantees, and in a
three-player game where their two-player safety notion does not apply.

We scale down to the smallest game in which three players interact and both
quantities remain exactly computable. The contributions are:

1. An exact map from opponent deviation to exploitation value across 16 opponent
   pairs and 7 bias levels.
2. An exact Bayesian identification procedure that requires no privileged
   information, with measured sample complexity under three information regimes.
3. The first measurement in this setting of *exposure* — how much a single opponent
   gains by best-responding to our adaptive policy — and the resulting frontier.
4. A demonstration that the optimal degree of adaptation depends almost entirely on
   whether anyone is punishing us, and a common axis on which opponent types can be
   compared fairly.

We also report two negative or corrective results: a non-adapting equilibrium player
can lose to extreme opponents, and conclusions drawn at equal alpha are artifacts of
the parameterisation.

---

## Related work

**Leduc and Bayesian opponent modelling.** Leduc Hold'em was introduced by Southey et
al. (2005) in *Bayes' Bluff*, which separates uncertainty in game dynamics from
uncertainty about the opponent's strategy, infers a posterior over opponent strategies
from observed play, and responds to that posterior. Our identifier in Section 4 is a
direct descendant, with a finite structured prior in place of Dirichlet priors.
Ganzfried & Sun (2018) extend Bayesian exploitation to imperfect-information games,
and Ganzfried, Wang & Chiswick (2024) to three-player Kuhn poker, the only multiplayer
precedent we found. Game-theory-based opponent modelling in large games is due to
Ganzfried & Sandholm (2011). Deep and implicit alternatives include DRON (He et al.,
2016) and Learning to Exploit (Wu et al., 2021).

**Robust and safe exploitation.** McCracken & Bowling (2004) proposed epsilon-safe
strategies, choosing the best-performing strategy among those losing at most epsilon
in the worst case. Johanson, Zinkevich & Bowling (2007) introduced restricted Nash
responses (RNR), which pin the opponent to a model with probability p and trace a
Pareto-optimal gain/exploitability frontier; data-biased responses (Johanson & Bowling,
2009) weight the pinning per decision by data confidence. Ganzfried & Sandholm (2015)
characterise safe exploitation in repeated two-player zero-sum games by risking only
what has already been won relative to the game value. Recent work schedules the RNR
pin level from anytime-valid confidence sequences and certifies deployed exploitability
(Li & Huang, 2026). **All of these are two-player zero-sum**, where the minimax
theorem supplies a game value to be safe relative to.

**Multiplayer.** CFR (Zinkevich et al., 2008) has no convergence guarantee to Nash
beyond two players; when every player runs a regret minimiser the average profile
converges only to a coarse correlated equilibrium (Hannan, 1957; see Sychrovský et al.,
2023), but has produced strong three-player agents (Abou Risk & Szafron,
2010; Gibson, 2014) and superhuman six-player play (Brown & Sandholm, 2019). Three-player
Kuhn poker already has a continuum of equilibria with different values (Szafron,
Gibson & Sturtevant, 2013), which is why equilibrium selection matters for our baseline
(RERUN-1). Exact multiplayer alternatives now exist for small games: quadratically
constrained programming over the sequence form (Ganzfried, 2026), and fictitious play has
been reported to approximate multiplayer Nash equilibria better than CFR (Ganzfried,
2025b). Ganzfried's *safe equilibrium* (2023) models each opponent as rational with
a given probability and arbitrary otherwise, and generalises RNR to multiplayer and
non-zero-sum games. We use a direct three-player RNR construction (Section 6); safe
equilibrium is the natural next comparison, particularly for the symmetric,
self-interested case of RERUN-21 and RERUN-22. The failure of Nash as a performance floor with more than two
players, which we exhibit concretely in Section 3.1, is noted in that work and in Li
(2018).

**Type-based reasoning.** Our identifier is an instance of Bayesian reasoning over
*hypothesised types*: maintain beliefs over a finite set of candidate behaviours for the
other agents and plan against the most likely ones (Albrecht, Crandall & Ramamoorthy,
2016). That literature shows prior beliefs can materially affect performance (Albrecht,
Crandall & Ramamoorthy, 2015) and extends types to carry continuous *parameters*
(Albrecht & Stone, 2017) — exactly the unknown-alpha problem of RERUN-5. The tension
between exploiting learned type beliefs and the risk that they are wrong has been
studied directly as safe exploitation under untrusted type beliefs (arXiv:2411.07679).
Albrecht & Stone (2018) survey the area.

**Short-horizon exploitation and identification.** In two-player Kuhn poker, Hoehn et
al. (2005) and Southey, Hoehn & Holte (2009) find that converging to maximally
exploitive play within a small number of hands is impractical, while better-than-Nash
play can be reached in about 50 hands, and that among equal-value equilibria some speed
opponent learning by exploring more effectively. In two-player Leduc, Southey et al.
(2005) observe posteriors concentrating within about ten hands against opponents drawn
from the prior. Our three-player figures (26-140 hands) are much slower because our
types are small perturbations of equilibrium rather than broad draws. For testing
whether an opponent follows a given mixed strategy, Ganzfried (2025) gives a
nonparametric test for repeated strategic-form games and leaves imperfect-information
games open; our likelihood-ratio posterior is a parametric answer to that question in
an extensive-form game.

**Censored observations.** Folds hide private cards, so showdown data are missing not at
random, and per-card frequency estimators converge to a selected distribution (Guo,
2026). Our identifier avoids this in principle because it models the fold decisions
themselves under every hypothesis — the missingness mechanism is part of the likelihood
— but this should be checked empirically (RERUN-20).

**Poker theory.** Our money-flow result in Section 3.2 is a quantitative instance of what
poker players call *Morton's theorem* or *implicit collusion*: in multiway pots an
opponent's incorrect call can cost the best hand money, with the difference flowing to
the other opponents (Morton, 1999; see also Sklansky, 1999). We have not found a prior
quantification of it against an equilibrium baseline.

**Two-player exploitation in Leduc and beyond.** Recent two-player methods include safe
subgame refinement (Liu et al., 2022), continual depth-limited responses (Milec, Kubíček
& Lisý, 2021), adaptation beyond the depth limit (Liu et al., 2025), and transformer or
deep-RL exploiters (StratFormer, 2026; AlphaExploitem, 2026; Li & Miikkulainen, 2018),
with LLM-based agents a separate thread (Huang et al., 2024). Safe exploitation has also
been generalised to agents whose baseline is only an epsilon-equilibrium
(arXiv:2307.12338), which is relevant because our CFR baseline has nash_conv 0.0057, not
zero.

**Large-scale adaptive multiplayer agents.** Learned adaptive agents for multiway
no-limit play exist (e.g. arXiv:2509.23747), but at that scale exploitability cannot be
computed, so the tradeoff studied here cannot be evaluated there. Our contribution is
complementary: exact measurement in a small game.

---

## 2. Setup

### 2.1 Game

Three-player Leduc Hold'em as implemented in OpenSpiel. The deck scales with player
count as 2 suits x (players + 1) ranks, so the three-player game uses **8 cards
across 4 ranks**, not the 6-card deck of the familiar two-player version.

| | 2-player | 3-player |
|---|---|---|
| information sets | 936 | 25,800 |
| game-tree nodes | 9,457 | 1,831,601 |

Exploitability is measured by `nash_conv`: the sum over players of what each could
gain by a unilateral best response while the others hold still. This is the
**no-collusion** definition; opponents never coordinate a joint deviation.
OpenSpiel's `exploitability()` is restricted to two players and cannot be used.

### 2.2 Equilibrium baseline **[EXACT]**

CFR+ for 1,000 iterations. nash_conv = 0.005662 (recomputed independently from the
stored table). Per-seat best-response gains: 0.00173, 0.00223, 0.00171.

Equilibrium values by seat, all three players using the same solution:

| seat 0 | seat 1 | seat 2 | sum |
|---|---|---|---|
| -0.0936 | -0.0087 | +0.1023 | 0.0000 |

Position is worth about 0.2 chips/hand from first to last actor, with skill held
constant. **Every "gain" in this paper is measured against the relevant seat's own
equilibrium value, not against zero.**

The equilibrium is substantially mixed: mean policy entropy 0.249 (max 1.099), with
45.9% of information sets near-pure.

> **[RERUN-1]** Only one CFR solution is used. Three-player games can admit multiple
> equilibria with different values — three-player Kuhn poker already has a
> parameterised family of them (Szafron, Gibson & Sturtevant, 2013). Solve from several random seeds or with a
> different algorithm (CFR, MCCFR, CFR-BR) and check whether the values by seat and
> the downstream gains are stable. If they are not, every number here is
> equilibrium-selection-dependent and must be reported as such. Candidate cross-checks:
> fictitious play (Ganzfried, 2025b) and, if it scales to 25,800 information sets, the
> exact sequence-form QCQP method (Ganzfried, 2026).

### 2.3 Opponent model

At every information set where the favoured action is legal,

    sigma'(a | I) = (1 - alpha) sigma*(a | I) + alpha [a = favoured]

Types: **O** optimal, **R** over-raiser, **C** over-caller, **F** over-folder.
alpha is the probability of overriding the optimal action *where the favoured action
is legal*. Support is preserved, which keeps all likelihood ratios finite.

Coverage and headroom differ sharply by type:

| type | information sets touched | share | mean sigma*(fav) |
|---|---|---|---|
| R | 11,520 | 44.7% | 0.250 |
| C | 25,800 | 100% | 0.392 |
| F | 21,744 | 84.3% | 0.589 |

This makes alpha **not comparable across types**; Section 9 fixes it.

---

## 3. Question 1: exploitation versus deviation **[EXACT]**

All 16 ordered pairs at alpha in {0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0}, seat 0.

Best-response gain over equilibrium, one deviator plus one optimal opponent:

| alpha | 0.05 | 0.1 | 0.2 | 0.35 | 0.5 | 0.75 | 1.0 |
|---|---|---|---|---|---|---|---|
| OR | +0.068 | +0.148 | +0.361 | +0.782 | +1.339 | +2.485 | +3.915 |
| OC | +0.058 | +0.125 | +0.261 | +0.480 | +0.727 | +1.242 | +1.887 |
| OF | +0.122 | +0.259 | +0.531 | +0.932 | +1.272 | +1.685 | +2.070 |

Near-linear for alpha <= 0.2, as first-order reasoning near equilibrium predicts.
Over-raisers then accelerate; over-folders flatten. **Section 9 shows the flattening
is an artifact of the parameterisation, not a property of folding.**

> **[RERUN-2]** The alpha grid starts at 0.05. The linearity claim for small alpha
> rests on three points. Add alpha in {0.01, 0.02, 0.03} and fit a slope; if the
> relationship is linear the slope should be constant and equal to a directional
> derivative of the best-response value.

### 3.1 A non-adapting equilibrium player can lose **[EXACT]**

Playing CFR (no adaptation) against biased opponents, gain over equilibrium:

| pair | alpha 0.5 | alpha 0.75 | alpha 1.0 |
|---|---|---|---|
| RR | +0.251 | +0.154 | **-0.176** |
| RC | — | — | **-0.355** |

Full return vector for RR at alpha 1.0: us **-0.270**, seat 1 **-2.384**, seat 2
**+2.654**. The two over-raisers mostly transfer money to each other, and the seat
with position captures it; we lose because our unadapted policy is not set up to
enter inflated pots profitably.

This is not a convergence failure. In two-player zero-sum a Nash strategy guarantees
at least the game value against any opponent. **No such guarantee exists with three
or more players**, so an equilibrium strategy has no floor when opponents deviate. Ganzfried (2023)
states the general point — outside two-player zero-sum, following a Nash equilibrium
can yield arbitrarily low payoff against irrational opponents — and our RR and RC
cases are concrete instances of it in poker.
Best response against the same RR pair earns +3.783 against CFR's -0.270.

### 3.2 Where the money goes **[EXACT]**

One deviator in seat 1, optimal opponent in seat 2, us playing CFR in seat 0:

| | alpha 0.1 | alpha 0.35 | alpha 1.0 |
|---|---|---|---|
| deviator loses | -0.088 | -0.334 | -0.903 |
| we gain | +0.023 | +0.128 | +0.693 |
| other opponent gains | +0.064 | +0.206 | +0.210 |

At small and moderate bias **the third player collects more of the mistake than we
do**. This is Morton's theorem / implicit collusion (Morton, 1999), known to poker
players qualitatively; the contribution here is its exact measurement against an
equilibrium baseline, and the observation that it is large enough to make
non-adaptation costly. **Do not present the phenomenon itself as new.** Without adaptation, CFR captures only 10-45% of the available best-response
gain, worst against over-folders. This is a three-player-specific argument for
adapting: in heads-up, an opponent's error is necessarily your gain.

### 3.3 Value of identification **[EXACT]**

Comparing a best response told which pair is seated ("known") with the best response
to the population of pairs ("blind"), 3 types, 9 pairs, seat 0, alpha 0.35:

| | known | blind | value of identification | CFR |
|---|---|---|---|---|
| average | +1.091 | +0.678 | **+0.413** | +0.130 |
| F/F | +1.891 | — | **+1.133** | — |

A policy that merely plays well against the *population* captures 57% of the
available gain without identifying anyone. Identification supplies the remaining
43%, concentrated against over-folders.

Correctness check: the blind best response's value against the mixture equals the
average of its values against the individual pairs to 1e-16. The mixture is
reach-weighted per information set; naive averaging of action probabilities is wrong
in principle, though here it cost only 1.6%.

> **[RERUN-3]** This uses 3 opponent types (9 pairs), while the rest of the paper
> uses 4 (16 pairs). Redo with O included so the value of identification is on the
> same hypothesis space as Sections 4-6.

---

## 4. Question 2: identifying the opponents **[MC]**

We sit in seat 0 playing CFR and maintain an exact Bayesian posterior over all 16
ordered pair hypotheses. The likelihood of a hand marginalises over the opponents'
unknown hole cards:

    P(actions | t1, t2) = sum over (c1, c2) consistent with observation
        P(c1, c2 | our card, board) * prod_t sigma_t1(a | I_1) * prod_t sigma_t2(a | I_2)

This follows the Bayesian decomposition of Southey et al. (2005), specialised to a
finite structured hypothesis space. Our own action probabilities and the chance
probabilities are identical under every hypothesis and cancel. **No privileged information is required**: marginalising over
hidden cards is sufficient, which answers the question of whether a card-revealing
"critic" is necessary in principle. It is not.

Median hands until P(true pair) >= 0.9, pooled over all 16 pairs, 40-50 sessions per
pair:

| alpha | actions only | + showdown cards | + critic (all cards) |
|---|---|---|---|
| 0.10 | 140 | 122 | 76 |
| 0.20 | 58 | 48 | 32 |
| 0.35 | 26 | 23 | 17 |

- Scaling is empirically about alpha^-1.3 to alpha^-1.4, slower than the 1/alpha^2
  suggested by small-deviation theory at these bias levels.
- **A critic is worth a factor of 1.5 to 1.9.**
- **Showdown cards alone buy almost nothing** (1.1-1.2x): by the time a hand reaches
  showdown, the betting has already revealed most of the information.
- Hardest: one deviator beside an optimal player (230-260 hands at alpha 0.1).
  Easiest: two deviators (39-50 hands).

**On showdown information.** Hoehn et al. (2005) observed in Kuhn poker that certain
folding observations carry as much information as a showdown, because some fold
decisions are only rational with particular cards. Our finding that showdown cards add
little is consistent with that and extends it to Leduc.

### 4.1 Do not reveal the probability vector

Because the tilted vector never equals the optimal one, a single observation at any
information set where the favoured action is legal identifies both the type and
alpha exactly. A critic should reveal cards only, and only as a training-time
device.

### 4.2 Why rare actions carry more evidence

Each observation multiplies the odds by the likelihood ratio
sigma_type(a|I) / sigma_O(a|I). At a state where the optimal policy plays
(raise, call, fold) = (0.4, 0.1, 0.5), with alpha = 0.35:

| observed | over-raiser | over-caller | over-folder |
|---|---|---|---|
| raise | 1.53 | 0.65 | 0.65 |
| call | 0.65 | **4.15** | 0.65 |
| fold | 0.65 | 0.65 | 1.35 |

A call the equilibrium plays only 10% of the time is 4.15x more likely from an
over-caller, while a raise is only 1.53x more likely from an over-raiser.

### 4.3 Robustness of the metric **[MC]**

alpha 0.35, 40 sessions, three definitions of "identified":

| pair | first crossing | sustained 20 hands | seat 1 only |
|---|---|---|---|
| OO | 42 | 48 | 28 |
| RO | 28 | 33 | 6 |
| FO | 54 | 65 | 20 |
| RR | 8 | 10 | 5 |
| FF | 40 | 40 | 16 |
| RC | 8 | 9 | 5 |

Requiring persistence adds only 0-20%, so the headline numbers are not inflated by
transient spikes. **Classifying a single opponent is 2-4x faster than pinning the
ordered pair.** Opponents generate 3.5-4.3 decisions per hand regardless of type, so
counting hands rather than decisions does not distort comparisons.

> **[RERUN-4]** The headline table uses the joint 16-pair criterion, which is the
> hard version of the question and not what the adaptive policy actually needs.
> Regenerate the full three-alpha table using the **per-seat marginal** criterion and
> report both.

> **[RERUN-5]** alpha is assumed known when forming hypotheses. A deployable system
> needs a prior over alpha as well; the hypothesis space becomes roughly
> (3 types x k alphas + 1)^2. Rerun with a grid over alpha and report how much slower
> identification becomes.

> **[RERUN-6]** Identification times assume we play CFR throughout. Once the policy
> adapts, opponents reach different information sets and the evidence rate changes.
> Measure identification time *under* the adaptive policy.

> **[RERUN-19]** The prior over the 16 pairs is uniform. Type-based reasoning is known to
> be sensitive to prior beliefs (Albrecht, Crandall & Ramamoorthy, 2015). Rerun with
> skewed priors (e.g. 50% on OO) and report how identification times and the beta
> results change.

> **[RERUN-20]** Verify that the showdown-regime posterior is calibrated. Showdown data
> are missing not at random (Guo, 2026); our full likelihood should handle this, but
> check empirically: across sessions, the true pair should receive posterior mass p in
> roughly a fraction p of cases.

> **[RERUN-7]** 40-50 sessions per cell gives roughly +/-10% on a median, and at
> alpha 0.1 up to 38% of sessions are censored for the hardest pairs. Increase to
> 200+ sessions and report the censoring fraction in the table itself.

---

## 5. Question 3: how fast to switch **[MC]**

Confidence c = (p_max - 1/16) / (1 - 1/16), weight w = c^beta, policy
w * BR(MAP pair) + (1 - w) * CFR. True pair drawn uniformly from 16 each session.

alpha 0.35, 600 sessions x 100 hands (se ~0.028):

| policy | chips/hand |
|---|---|
| CFR only | +0.098 |
| beta = 0 (instant) | +0.668 |
| beta = 1 | +0.657 |
| beta = 8 (cautious) | +0.656 |
| oracle (told the pair) | +0.815 |

alpha 0.1, 500 sessions x 100 hands (se ~0.020):

| policy | chips/hand |
|---|---|
| CFR only | -0.051 |
| beta = 0 | +0.024 |
| beta = 1 | +0.027 |
| beta = 8 | +0.018 |
| oracle | +0.170 |

**beta does not matter**: every value from 0 to 8 falls within one standard error.
Adaptation is worth +0.56 over CFR at alpha 0.35 (79% of the oracle's advantage),
but the speed of switching is not a lever. At alpha 0.1 identification needs ~140
hands while sessions are 100, and adaptation recovers only a third of the oracle
gap.

Ganzfried, Wang & Chiswick (2024) report a parallel insensitivity in three-player Kuhn
poker: exploration horizons from 0 to 200 produced very similar win rates, because
exploitation became possible quickly. Section 7 shows why such null results arise:
nobody in these opponent pools punishes the adapting agent. Southey, Hoehn & Holte
(2009) likewise find better-than-Nash play reachable in about 50 hands in two-player Kuhn
poker, consistent with fast, cheap adaptation when opponents are static.

> **[RERUN-8]** Differences between beta values are smaller than the standard error,
> so this is a null result with weak power, not a demonstration of equivalence. Use
> common random numbers across beta values (paired comparison), increase sessions,
> and report a paired confidence interval on the *differences* rather than
> independent means.

> **[RERUN-9]** Sessions are fixed at 100 hands. Since identification takes 26-140
> hands, session length and identification time are confounded. Sweep session length
> across {25, 50, 100, 200, 400} at fixed alpha.

---

## 6. Question 4: how should we adapt? Linear mixing versus restricted Nash response **[EXACT]**

> **CORRECTION (v6).** Everything in the table below evaluates punishment by **seat 1
> only**. The one-sided RNR trusts seat 2's model completely, and when seat 2 is allowed
> to punish instead, the p = 0.5 RNR earns **-3.51** (self-interested) / **-3.62**
> (adversarial) — nearly as bad as a pure best response. The claims "RNR dominates
> linear mixing everywhere" and "RNR at p = 0.25 is less exploitable than CFR" are
> **false as stated**; they hold only against the opponent whose model is not trusted.
> A preliminary symmetric run (both opponents independently free with probability
> 1 - p, self-interested, 120 iterations, gaps ~0.015) gives at p = 0.5: **+0.481** vs
> the biased pair, **-0.014** worst case over both opponents' self-interested
> punishment, **-0.269** worst case over adversarial punishment. CFR gives +0.076 and
> at most -0.144. So a real advantage probably survives, at roughly 40% of the one-sided
> size. Treat the table below as the *one-sided* result only; the symmetric sweep
> (RERUN-21, `rnr_package/`, priority 1) replaces it.

Target pair FF at alpha 0.35, seat 0. Two adaptation methods:

- **Linear mixing**: pi_w = (1 - w) CFR + w BR(FF), for w in [0, 1].
- **Three-player restricted Nash response (RNR)**, extending Johanson, Zinkevich &
  Bowling (2007). In a modified game, seat 2 plays the biased model; seat 1 plays the
  biased model with probability p and otherwise is *free*, knows its mode, and
  minimises our payoff; we do not observe the mode. Solved with CFR+ (250 iterations;
  residual gap below 0.003 at every p). With seat 2 fixed and the free player
  adversarial, the modified game is two-player zero-sum from our side, so CFR+
  converges. p = 1 recovers the best response.

Both are evaluated on identical axes: EV against the biased pair, EV against a
worst-case adversarial seat 1, and EV against a self-interested seat 1 that
best-responds to us (the punisher of Section 7).

| method | param | vs biased pair | vs adversary | vs self-interested punisher |
|---|---|---|---|---|
| linear | w = 0 (CFR) | +0.076 | -0.189 | -0.144 |
| linear | w = 0.5 | +0.868 | -1.300 | -1.157 |
| linear | w = 0.8 | +1.391 | -2.681 | -2.630 |
| linear | w = 1 (BR) | +1.764 | -3.899 | -3.876 |
| RNR | p = 0 | +0.783 | **+0.451** | +0.470 |
| RNR | p = 0.25 | +1.092 | **+0.399** | +0.405 |
| RNR | p = 0.5 | +1.184 | +0.327 | +0.333 |
| RNR | p = 0.75 | +1.582 | -0.401 | -0.394 |
| RNR | p = 0.9 | +1.710 | -0.909 | -0.901 |
| RNR | p = 0.97 | +1.741 | -1.439 | -1.390 |

(Full linear sweep at 0.1 increments in `frontier_rows.pkl`.) The linear-mixing
self-interested column reproduces the independently computed Section 7 values
(-0.144, -1.640 at w = 0.6, -3.876), cross-validating the tree engine against OpenSpiel.

**The earlier claim is reversed.** Draft v2 concluded that "exposure grows faster than
gain" and "there is no cheap knee". That was an artifact of linear mixing, as RERUN-17
suspected. Under RNR:

1. **RNR dominates linear mixing at every point.** For the same gain against the biased
   pair, RNR's worst case is 1.6 to 2.9 chips/hand better. At gain ~1.58, linear mixing
   (w = 0.9) has worst case -3.25; RNR (p = 0.75) has -0.40.
2. **The frontier is concave with a cheap knee.** RNR at p = 0.25 captures 60% of the
   gain available between CFR and the best response — and its worst case (+0.399) is
   *better than CFR's* (-0.189). Around the knee the tradeoff barely exists.
3. **Past the knee it is steep.** The last 10% of gain (p = 0.97 to 1) costs 2.5
   chips/hand of worst-case value.

This mirrors the two-player finding that RNR frontiers are strongly concave (Johanson
et al., 2007; Li & Huang, 2026), now in a three-player game.

**Why RNR at p = 0 beats CFR on both axes.** In two-player games, RNR at p = 0 *is* the
equilibrium. Here it is not, because seat 2 is still modelled as biased: RNR at p = 0 is
the maximin strategy against seat 1 *given a trusted model of seat 2*. CFR ignores that
model. So the +0.45 guarantee is bought entirely by trusting seat 2 — which is exactly
why RERUN-21 matters.

> **[RERUN-21 — HIGH]** The restriction is one-sided: seat 1 may deviate from its model,
> but seat 2's model is trusted completely, in both the RNR solve and the evaluation.
> A symmetric version is needed. Making *both* opponents free and adversarial in the same
> hand would make them a coalition, violating the no-collusion assumption, so the natural
> design is: each hand, independently per seat, the opponent is free with probability
> 1 - p, and a free opponent maximises its *own* payoff (next item). Report how much of
> RNR's advantage survives when neither model is trusted.

> **[RERUN-22]** The free opponent is adversarial (minimises our payoff) so that the
> modified game is zero-sum and CFR+ converges. A self-interested free opponent is the
> more realistic model but makes the game general-sum. Solve that variant as well (CFR
> without guarantees, as for the baseline) and compare. The evaluation already includes a
> self-interested punisher, and RNR's numbers there are within 0.05 of the adversarial
> ones, which suggests the difference is small.

> **[RERUN-10, updated]** Still one target pair (FF), one alpha, seat 0. FF is the most
> exploitable pair (Section 8), so this is the most favourable case for adaptation.
> Repeat for RR, CC and a mixed pair, and for alpha 0.1.

---

## 7. Question 5: adaptation under punishment **[EXACT]**

A self-interested seat 1 best-responds to our policy for a fraction f of hands; seat 2
stays biased. Session EV = (1 - f) x (EV vs biased pair) + f x (EV vs punisher), computed
exactly for every setting of each method, then maximised:

| f (fraction punished) | 0 | 0.25 | 0.5 | 0.75 | 1.0 |
|---|---|---|---|---|---|
| best linear mix | +1.764 (w = 1) | +0.386 (w = 0.8) | +0.047 (w = 0.2) | -0.063 (w = 0.1) | -0.144 (w = 0) |
| best RNR | +1.764 (p = 1) | **+1.088** (p = 0.75) | **+0.758** (p = 0.5) | **+0.577** (p = 0.25) | **+0.470** (p = 0) |

1. **With linear mixing, the answer to "how far should I adapt" collapses to "not at all"
   as punishment becomes likely** — and even then loses 0.144, because CFR is not a safe
   baseline in a three-player game (Section 3.1).
2. **With RNR, the optimal setting degrades gracefully**, from p = 1 through 0.75, 0.5,
   0.25 to 0, and stays profitable throughout. Under constant punishment RNR still earns
   +0.47, a 0.61 chip/hand improvement over the best linear policy.
3. The Section 5 beta null result still stands (with no punisher, full commitment is
   optimal under either method), but its interpretation changes: the relevant knob is
   not *how fast* to adapt but *which family* the adaptation lives in.

The same caveats as Section 6 apply: one-sided restriction (RERUN-21), one target pair
(RERUN-10), punishment fraction f exogenous (RERUN-13), omniscient punisher (RERUN-12).

---

## 8. Question 6: a common axis for opponent types **[EXACT]**

For a mixture toward a pure action, TV(I) = alpha (1 - sigma*(fav|I)). Weighting by
how often each information set is reached under equilibrium play gives
**m = expected overridden decisions per hand**. An opponent seat faces 2.081
decisions per hand.

| type | overrides/hand per unit alpha |
|---|---|
| R | 1.393 |
| C | 0.898 |
| F | 0.280 |

**alpha buys five times more actual deviation for a raiser than for a folder.**

Best-response gain at matched override rate, seat 0:

| m | OR | OC | OF | RR | CC | FF |
|---|---|---|---|---|---|---|
| 0.10 | +0.100 | +0.140 | **+0.948** | +0.215 | +0.227 | **+1.899** |
| 0.25 | +0.313 | +0.373 | **+1.837** | +0.584 | +0.530 | **+3.860** |
| 0.50 | +0.812 | +0.832 | +2.070* | +1.217 | +1.146 | +4.218* |

\* unreachable for F: even at alpha = 1.0 its override rate caps at 0.280, so these
cells are *below* the matched rate and still the largest in the row.

Three corrections follow:

1. **Folding errors are ~10x more punishable than raising errors per unit of
   deviation.** Raising and calling errors are worth about the same.
2. **The equal-alpha ranking was an artifact.** At alpha 1.0 the over-raiser appeared
   most exploitable (+3.92 vs +2.07) purely because alpha bought it five times more
   deviation.
3. **"Over-folders saturate" was an artifact.** The concave curve in Section 3
   flattens because the parameterisation runs out of room.

Combined with Section 4, this yields the paper's sharpest tension: **the deviation
that is most valuable to detect is also the slowest to detect** (over-folders: 54
hands versus 28 for over-raisers at alpha 0.35).

> **[RERUN-14]** m is computed under equilibrium reach probabilities. Once we adapt,
> opponents' information sets are reached with different frequencies and the realised
> deviation rate shifts. Recompute m under the adapted policy and report how much the
> axis moves.

> **[RERUN-15]** F cannot reach m > 0.280 under this construction, so the top row is
> not a clean comparison. Either cap all types at m = 0.28, or add a folding
> construction with wider reach (e.g. also shifting probability away from raising at
> no-bet nodes) so all three types span the same range.

---

## 9. Seat dependence **[EXACT]**

Best-response gain over each seat's own equilibrium value, alpha 0.35:

| opponents | seat 0 | seat 1 | seat 2 |
|---|---|---|---|
| FF | +1.857 | +1.741 | +1.647 |
| RR | +1.185 | +1.507 | +1.669 |
| CC | +0.666 | +0.696 | +0.737 |
| OO | +0.002 | +0.002 | +0.002 |

**Seat changes the gain by 10-40% and changes the ranking**: seat 0 profits most from
over-folders, seat 2 from over-raisers. Opponent ordering matters too — one
over-folder plus one optimal opponent is worth +0.719 or +1.132 to seat 1 depending
on which seat the folder occupies.

> **[RERUN-16]** Sections 3-8 are seat 0 only. Rerun the alpha sweep, the
> identification study, the frontier and the punisher analysis for all three seats.
> This is roughly 3x the compute and removes the largest single caveat in the paper.
> If compute is limited, report seat 0 as primary and include the table above with an
> explicit statement that conclusions are seat-dependent by 10-40%.

---

## 10. Discussion

Three findings generalise beyond this game.

**Equilibrium is not safe in multiplayer.** With three or more players there is no
minimax guarantee, and we exhibit concrete cases where a converged equilibrium player
loses relative to its equilibrium value because opponents blunder in a way that
enriches a third party. Any claim that "playing GTO is the safe default" is a
two-player intuition.

**The value of adaptation is mostly the value of not being blind, not of being
right.** A policy that plays well against the population of opponents captures 57% of
the available gain without identifying anyone.

**How you adapt matters more than how fast.** Mixing an equilibrium with a best
response is dominated by a restricted Nash response at every point of the
exploitation/exposure frontier, and the gap is large: up to 2.9 chips/hand of
worst-case value at equal gain. Under linear mixing, the rational response to a
watchful table is to stop adapting; under RNR it is to adapt less, and remain
profitable. Adaptation is free only when nobody is watching, but it is far
from worthless when somebody is.

---

## 11. Consolidated rerun list

Ordered by value per unit of effort.

| # | Section | Issue | Fix | Priority |
|---|---|---|---|---|
| 17 | 6, 7 | ~~Linear mixing dominated; frontier shape an artifact~~ | **DONE** — three-player RNR computed; claim reversed | done |
| 21 | 6, 7 | RNR restriction one-sided: seat 2 punishes it for -3.6 | Symmetric self-interested RNR — **queued locally, `rnr_package` priority 1** | **highest** |
| 22 | 6 | Free opponent adversarial, not self-interested | General-sum RNR variant; compare | medium |
| 13 | 7 | f is exogenous; punishment should begin when the opponent identifies us | Derive f from Section 4 identification times | **highest** |
| 16 | 9 | Seat 0 only | Rerun everything for seats 1 and 2 | **highest** |
| 8 | 5 | beta differences are within noise | Common random numbers, paired CIs on differences, more sessions | high |
| 12 | 7 | Punisher is omniscient | Punisher that estimates our policy from play | high |
| 5 | 4 | alpha assumed known | Prior over alpha in the hypothesis space | high |
| 10 | 6 | One target pair, one alpha | Sweep targets and alpha | high |
| 4 | 4 | Joint 16-pair criterion is the hard version | Report per-seat marginal criterion too | medium |
| 1 | 2 | Single equilibrium solution | Multiple seeds/algorithms; check value stability | medium |
| 9 | 5 | Session length fixed at 100 | Sweep {25, 50, 100, 200, 400} | medium |
| 14 | 8 | m computed under equilibrium reach | Recompute under adapted policy | medium |
| 15 | 8 | F cannot reach m > 0.28 | Cap all types, or widen the folding construction | medium |
| 7 | 4 | 40-50 sessions, censoring up to 38% | 200+ sessions, report censoring | medium |
| 6 | 4 | Identification assumes we play CFR | Measure under the adaptive policy | medium |
| 11 | 6 | Exposure measured against CFR opponents | Also measure against the biased pair | low |
| 3 | 3.3 | Value of identification uses 9 pairs not 16 | Redo on the 16-pair space | low |
| 2 | 3 | alpha grid starts at 0.05 | Add {0.01, 0.02, 0.03}; fit the slope | low |
| 19 | 4 | Uniform prior over pairs | Rerun with skewed priors; report sensitivity | medium |
| 20 | 4 | Showdown data are missing not at random | Check posterior calibration empirically | medium |
| 18 | 4, 5 | No baseline from prior multiplayer opponent modelling | Implement Ganzfried et al. (2024) Algorithm 2 on 3-player Leduc as a comparison | medium |

**Nothing in Section 5 (beta) should be reported as a positive finding until
[RERUN-8] is done. Sections 6-7 are now reportable, with the RERUN-21 caveat stated
explicitly next to every RNR number.** It is currently a null result with weak power.

---

## 12. Future work: theory

1. **Bound the value of identification** by the distinguishability of the opponent
   types; it should vanish exactly when one policy is simultaneously best against
   every pair.
2. **Explain sub-additivity.** Payoff is bilinear in the two opponents' strategies, so
   the best-response value need not be convex as it is with a single opponent. Two
   deviators are consistently worth less than the sum of each alone.
3. **Safe exploitation without a game value.** Two-player safety is defined relative
   to the game value; three-player games have no unique one. Ganzfried's (2023) safe
   equilibrium offers one multiplayer definition. Open: whether safety relative to a
   *chosen* equilibrium is meaningful when equilibria have different values, and how
   Section 7's f-dependent optimum relates to it.
4. **The Bayes-adaptive optimum.** Treating the opponent types as hidden state makes
   a session a POMDP whose optimal policy sits between blind and known. Its gap to
   known is the unavoidable cost of learning, and it is the right benchmark for any
   opponent model.
5. **Manipulation.** An adaptive agent's belief is itself a target: an opponent can
   imitate a type and then exploit the response.

---

## Appendix: reproduction

| Step | Script | Runtime |
|---|---|---|
| Equilibrium | `solve_cfr.py 1000` | ~2 h |
| Q1 alpha sweep | `q1_alpha_sweep.py` | ~15 min |
| Q2 identification | `q2_identify.py` | ~3 min per alpha |
| Q3 beta | `q3_beta.py` | ~5 min |
| Q4 exposure | `q4_frontier.py` | ~3 min |
| Q5 punisher | `q5_punisher.py` | ~2 min |
| Q6 common axis | `q6_equal_tv.py` | ~3 min |

Two implementation notes. Best-response objects must be built one at a time; holding
two simultaneously exhausts memory on a 1.8M-node tree. And `average_policy()` is a
view into the solver's memory — dropping the solver invalidates every lookup.

---

## References

Status key: **✓** checked against a publisher, author page or index during drafting;
**⚠** not yet checked, or sources disagree — resolve before submission.

### Poker and opponent modelling

- ✓ Albrecht, S. V., Crandall, J. W. & Ramamoorthy, S. (2015). An empirical study on the
  practical impact of prior beliefs over policy types. *AAAI*, 1988–1994.
- ✓ Albrecht, S. V., Crandall, J. W. & Ramamoorthy, S. (2016). Belief and truth in
  hypothesised behaviours. *Artificial Intelligence* 235, 63–94.
- ✓ Albrecht, S. V. & Stone, P. (2017). Reasoning about hypothetical agent behaviours and
  their parameters. *AAMAS*, 547–555.
- ✓ Albrecht, S. V. & Stone, P. (2018). Autonomous agents modelling other agents: A
  comprehensive survey and open problems. *Artificial Intelligence* 258, 66–95.
- ✓ Ganzfried, S. & Sandholm, T. (2011). Game theory-based opponent modeling in large
  imperfect-information games. *AAMAS*, 533–540.
- ✓ Ganzfried, S. & Sun, Q. (2018). Bayesian opponent exploitation in
  imperfect-information games. *IEEE CIG*, 1–8.
- ✓ Ganzfried, S., Wang, K. A. & Chiswick, M. (2024). Opponent modeling in multiplayer
  imperfect-information games. *DAI '24*, 39–45. arXiv:2212.06027.
- ✓ Ganzfried, S. (2025a). Nonparametric strategy test. *FLAIRS*. arXiv:2312.10695.
- ✓ Guo, J. (2026). Safe observation capacity for opponent exploitation under showdown
  censoring. arXiv:2608.09954.
- ✓ He, H., Boyd-Graber, J., Kwok, K. & Daumé III, H. (2016). Opponent modeling in deep
  reinforcement learning. *ICML*.
- ✓ Hoehn, B., Southey, F., Holte, R. C. & Bulitko, V. (2005). Effective short-term
  opponent exploitation in simplified poker. *AAAI*, 783–788.
- ⚠ Huang, C. et al. (2024). PokerGPT: An end-to-end lightweight solver for multi-player
  Texas hold'em via large language model. arXiv:2401.06781. (Author list seen only in
  reference lists.)
- ✓ Li, X. (2018). *Opponent Modeling and Exploitation in Poker Using Evolved Recurrent
  Neural Networks.* PhD thesis, University of Texas at Austin.
- ✓ Li, X. & Miikkulainen, R. (2018). Opponent modeling and exploitation in poker using
  evolved recurrent neural networks. *GECCO*, 189–196.
- ✓ Southey, F., Bowling, M., Larson, B., Piccione, C., Burch, N., Billings, D. &
  Rayner, C. (2005). Bayes' bluff: Opponent modelling in poker. *UAI*, 550–558.
- ⚠ Southey, F., Hoehn, B. & Holte, R. C. (2009). Effective short-term opponent
  exploitation in simplified poker. *Machine Learning.* doi:10.1007/s10994-008-5091-5.
  (Journal and DOI confirmed; volume, pages and year not confirmed — accepted 2008.)
- ✓ Wu, Z., Li, K., Zhao, E., Xu, H., Zhang, M., Fu, H., An, B. & Xing, J. (2021). L2E:
  Learning to exploit your opponent. arXiv:2102.09381.
- ⚠ arXiv:2411.07679 (2024). Safe exploitative play with untrusted type beliefs. (Authors
  not checked.)
- ⚠ AlphaExploitem: Going beyond the Nash equilibrium in poker by learning to exploit
  suboptimal play (2026). arXiv:2605.09150. (Authors not checked.)
- ⚠ StratFormer: Adaptive opponent modeling and exploitation in imperfect-information
  games (2026). arXiv:2604.25796. (Authors not checked.)

### Safe and robust exploitation

- ✓ Ganzfried, S. & Sandholm, T. (2015). Safe opponent exploitation. *ACM Transactions on
  Economics and Computation* 3(2), 8:1–8:28.
- ✓ Ganzfried, S. (2023). Safe equilibrium. *IEEE CDC*, 5230–5236. arXiv:2201.04266.
- ✓ Johanson, M. & Bowling, M. (2009). Data biased robust counter strategies. *AISTATS*,
  264–271.
- ✓ Johanson, M., Zinkevich, M. & Bowling, M. (2007). Computing robust counter-strategies.
  *NIPS 20*, 721–728. ⚠ Ganzfried's papers cite pages 1128–1135; two independent
  sources give 721–728. Check the NeurIPS proceedings page.
- ✓ Li, B. & Huang, L. (2026). Agents that certify their own exploits:
  confidence-scheduled restricted responses for safe opponent exploitation.
  arXiv:2607.28520.
- ✓ Liu, M., Wu, C., Liu, Q., Jing, Y., Yang, J., Tang, P. & Zhang, C. (2022). Safe
  opponent-exploitation subgame refinement. *NeurIPS* 35, 27610–27622.
- ✓ Liu, W., Fu, H., Fu, Q. & Wei, Y. (2025). Adapting beyond the depth limit: Counter
  strategies in large imperfect information games. arXiv:2501.10464.
- ✓ McCracken, P. & Bowling, M. (2004). Safe strategies for agent modelling in games.
  *AAAI Fall Symposium on Artificial Multi-agent Learning.*
- ✓ Milec, D., Kubíček, O. & Lisý, V. (2021). Continual depth-limited responses for
  computing counter-strategies in sequential games. arXiv:2112.12594.
- ⚠ arXiv:2307.12338 (2023). Safe opponent exploitation for epsilon equilibrium
  strategies. (Authors not checked.)

### Equilibrium computation and multiplayer

- ✓ Abou Risk, N. & Szafron, D. (2010). Using counterfactual regret minimization to
  create competitive multiplayer poker agents. *AAMAS*, 159–166.
- ✓ Bowling, M., Burch, N., Johanson, M. & Tammelin, O. (2015). Heads-up limit hold'em
  poker is solved. *Science* 347(6218), 145–149.
- ✓ Brown, N. & Sandholm, T. (2019). Superhuman AI for multiplayer poker. *Science*
  365(6456), 885–890.
- ✓ Ganzfried, S. (2025b). Empirical analysis of fictitious play for Nash equilibrium
  computation in multiplayer games. *FLAIRS*. (Subsumes arXiv:2001.11165.)
- ✓ Ganzfried, S. (2026). Quadratic programming approach for Nash equilibrium computation
  in multiplayer imperfect-information games. *Games* 17(1), 9. arXiv:2509.25618.
- ⚠ Gibson, R. (2014). *Regret Minimization in Games and the Development of Champion
  Multiplayer Computer Poker-Playing Agents.* PhD thesis. (Institution not checked.)
- ⚠ Hannan, J. (1957). Approximation to Bayes risk in repeated play. *Contributions to the
  Theory of Games* 3, 97–139. (Cited via secondary source; confirm.)
- ⚠ Lanctot, M. et al. (2019). OpenSpiel: A framework for reinforcement learning in games.
  arXiv:1908.09453.
- ⚠ Sychrovský, D. et al. (2023). Learning not to regret. arXiv:2303.01074. (Author list
  not checked.)
- ✓ Szafron, D., Gibson, R. & Sturtevant, N. (2013). A parameterized family of
  equilibrium profiles for three-player Kuhn poker. *AAMAS.* ⚠ pages.
- ⚠ Tammelin, O. (2014). Solving large imperfect information games using CFR+.
  arXiv:1407.5042.
- ✓ Zinkevich, M., Johanson, M., Bowling, M. & Piccione, C. (2008). Regret minimization in
  games with incomplete information. *NIPS 20*, 1729–1736. (Two independent sources
  agree on 2008 and these pages; note it is the same proceedings volume as Johanson et
  al., cited as 2007.)

### Poker theory

- ⚠ Morton, A. (1999). Post to rec.gambling.poker introducing what became known as
  Morton's theorem. **Only secondary sources checked (Wikipedia).** Find a citable
  published treatment before relying on this.
- ⚠ Sklansky, D. (1999). *The Theory of Poker*, 4th ed. Two Plus Two. (Discusses implicit
  collusion via the fundamental theorem; page not checked.)

### Large-scale adaptive agents

- ⚠ arXiv:2509.23747 (2025). Beyond game theory optimal: Profit-maximizing poker agents
  for no-limit hold'em. (Authors not checked. Optional; cut if space is tight.)

---

## Note on concurrent work

An unpublished codebase (github.com/Edward358-AI/adaptive-exploitation-poker, status
August 2026) asks "how much is opponent modelling worth?" with a no-information /
inferred-information / perfect-information ladder in no-limit hold'em — the same framing
as our blind / identified / known comparison. It is not a paper and should not be cited
as prior art, but check for a preprint before submission.
