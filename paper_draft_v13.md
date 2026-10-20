# Safe Exploitation Without a Game Value: Restricted Responses in Three-Player Leduc Hold'em

**Draft v13.** Adds restricted responses over a grid of independent biases, with an adaptive search over trust (§7.10); pessimistic re-evaluation and stability re-solves for that section are running. v12 added independent biases (§4.6, §5.2), a continuous-bias identifier (§6.2) and identification with adaptation in play (§8).

Tags: **[EXACT]** exact game-tree evaluation on the 5,000-iteration baseline.
**[MC]** Monte Carlo.
**[PRELIM]** few sessions; indicative, not yet established.
All payoffs are chips per hand, measured relative to the seat's own equilibrium value unless
stated otherwise.

---

## Abstract

In two-player zero-sum games, safe opponent exploitation is well understood: restricted Nash
responses trace the frontier between profiting from an opponent's mistakes and becoming
exploitable, and safety is measured against the game's value. Neither tool carries over to three
or more players, where there is no unique game value and an equilibrium strategy guarantees
nothing. We study the problem exactly in three-player Leduc Hold'em, against opponents built by
biasing a converged equilibrium toward raising, calling or folding, each with its own degree of
bias. We find: (i) equilibrium play is not a floor — a fixed equilibrium strategy loses up to 0.69
chips/hand to extreme opponents, an almost-indifferent opponent can shift up to 0.23 chips/hand
between the other two players, two self-interested opponents who keep learning converge to an
equilibrium that costs a fixed equilibrium player 0.39 chips/hand, and equilibrium play becomes
punishable mainly when an over-raiser is at the table; (ii) a three-player restricted Nash response
that trusts neither opponent's model achieves **safe exploitation** — at least 0.05 chips/hand more
profit with no material loss of worst-case safety — in 41 of 45 settings at moderate bias (median
extra profit 0.33), 9 of 12 at intermediate bias and 8 of 15 at mild bias, with an abrupt switch
from a safe to an exposed regime as trust grows — figures that a finer search over trust suggests
are conservative: over a grid of independent biases, safe exploitation holds in 177 of 189
settings (median extra profit 0.75); and (iii) worst-case safety is far more sensitive
than profit to small strategy differences and to how a near-indifferent punisher breaks ties. First
sessions joining Bayesian identification with adaptation in play show adaptation working against
opponents the model describes — but when all three players adapt, every player becomes certain of
a wrong model within 50 hands: confidence measures certainty within the model, not whether it fits.

---

## 1. Introduction

An equilibrium strategy cannot be exploited but leaves money on the table against weak
opponents; a best response wins the most against a specific opponent but can be exploited in
turn. In two-player zero-sum games this tradeoff has a well-developed theory. Restricted Nash
responses (Johanson, Zinkevich & Bowling, 2007) trace its Pareto frontier, epsilon-safe
strategies bound worst-case loss (McCracken & Bowling, 2004), and safe exploitation risks only
what has already been won relative to the game value (Ganzfried & Sandholm, 2015).

Every one of those guarantees rests on the minimax theorem. With three players there is no game
value to be safe relative to, an equilibrium strategy guarantees nothing, and one opponent's
mistake need not become your gain. The tradeoff is therefore not just unmeasured in multiplayer
poker but not yet well defined. Measuring it requires a game small enough for exact best
responses and exploitability, and large enough that three players genuinely interact.

Three-player Leduc Hold'em is that game. Our contributions:

1. **Equilibrium is not a floor.** Concrete cases where a converged equilibrium player loses to
   deviating opponents, a measurement of where their mistakes' value goes, and a second
   equilibrium — reached by independently learning opponents — that costs a fixed equilibrium
   player 0.39 chips/hand.
2. **A three-player restricted Nash response** with symmetric, self-interested restriction, and
   an exact evaluation across 15 ordered opponent pairs, three seats and three bias levels.
3. **A sensitivity analysis of the worst-case measure.** Worst-case safety moves by up to 0.1
   chips/hand under strategy changes that leave profit unchanged; we calibrate comparisons to
   this noise and to punisher tie-breaking.
4. **Structure of the safe region**: where it lies, which opponent is dangerous to trust, and how
   robust it is to modelling errors.
5. **Independent biases**: the value of exploitation, the safety of equilibrium play, and safe
   exploitation mapped over grids of separate biases for the two opponents.
6. **Identification and adaptation in play** (preliminary): an identifier with a continuous bias
   per opponent, and a simulator in which any player may adapt to a moving posterior.

---

## 2. Related work

**Robust and safe exploitation.** Restricted Nash responses pin the opponent to a model with
probability p and otherwise let it best-respond, tracing a Pareto-optimal gain/exploitability
frontier (Johanson et al., 2007); data-biased responses weight the pinning by data confidence
(Johanson & Bowling, 2009). Epsilon-safe strategies (McCracken & Bowling, 2004) and safe
opponent exploitation (Ganzfried & Sandholm, 2015) bound risk relative to the game value.
All are two-player zero-sum. Ganzfried's safe equilibrium
(2023) generalises restricted responses to multiplayer and non-zero-sum games and is the closest
theoretical relative of our construction.

**Opponent modelling.** Leduc Hold'em was introduced with Bayesian opponent modelling (Southey et
al., 2005); short-horizon exploitation in Kuhn poker was studied by Hoehn et al. (2005) and
Southey, Hoehn & Holte (2009). Our identifier is Bayesian reasoning over hypothesised types
(Albrecht, Crandall & Ramamoorthy, 2016), including parameterised types (Albrecht & Stone, 2017;
survey in Albrecht & Stone, 2018). The only multiplayer precedent we found is Ganzfried, Wang &
Chiswick (2024), who model opponents in three-player Kuhn poker. Showdown data are missing not at
random — whether a hand reaches showdown depends on the very behaviour being modelled; our
likelihood, like Southey et al.'s, models fold decisions explicitly and so handles this in principle.

**Multiplayer equilibrium.** CFR (Zinkevich et al., 2008) has no Nash guarantee beyond two
players, though it has produced strong three-player (Abou Risk & Szafron, 2010) and superhuman
six-player agents (Brown & Sandholm, 2019). Three-player Kuhn poker already has a parameterised
family of equilibria (Szafron, Gibson & Sturtevant, 2013), which foreshadows the equilibrium
selection effects we measure. The observation that an opponent's mistake can benefit a third player is poker folklore,
sometimes called Morton's theorem or implicit collusion; we quantify it against an equilibrium baseline.

---

## 3. Setup

### 3.1 Game

Three-player Leduc Hold'em in OpenSpiel (Lanctot et al., 2019). The deck scales with player
count as 2 suits × (players + 1) ranks, so the three-player game uses 8 cards across 4 ranks.
It has 25,800 information sets and 1,831,601 game-tree nodes, small enough for exact
evaluation.

### 3.2 Equilibrium baseline **[EXACT]**

CFR+ for 5,000 iterations; nash_conv 0.0023, with each seat's unilateral best-response gain
below 0.001. Equilibrium values by seat: **−0.0949, −0.0089, +0.1038** (seat 0 acts first).
Position is worth about 0.2 chips/hand from first to last actor.

### 3.3 Opponents

Each opponent tilts the equilibrium toward one action, wherever that action is legal:

    sigma'(a | I) = (1 - alpha) sigma*(a | I) + alpha [a = favoured]

Types: **O** equilibrium, **R** over-raiser, **C** over-caller, **F** over-folder. Pairs are
written in seat order of the two opponents (for a learner in seat 0, `FR` means seat 1 folds too
much and seat 2 raises too much); all 16 ordered pairs are used, with OO as a control.

alpha is an override rate where the favoured action is legal, so it is not comparable across
types: raising is illegal at the cap and folding is illegal with no bet to face. Per unit of
alpha, an over-raiser deviates on 1.41 decisions per hand, an over-caller on 0.89, an
over-folder on only 0.28.

**Independent biases.** In §4.6, §5.2, §6.2 and §8 each opponent has its own alpha, on a grid
0, 0.05, …, 0.6. At alpha = 0 every type *is* the equilibrium player, so such tables are shared
between pairs.

### 3.4 Metrics

- **Gain:** our expected payoff against the biased pair, minus our seat's equilibrium value.
- **Worst case:** our expected payoff if *either* opponent abandons its model and plays the
  exact best response to our strategy, maximising its *own* payoff, while the other keeps
  playing its model; we take the worse of the two, minus our equilibrium value. The switching
  opponent is omniscient about our strategy, so this bounds a realistic punisher. An adversarial
  variant (minimising our payoff) also measures spite and is reported only as a bound.
- **Baseline:** the best of three equilibrium-level strategies, on each axis separately — our
  5,000-iteration CFR table, the same run at 1,000 iterations, and the p = 0 restricted response
  (§4.5).
- **Safe exploitation:** some p > 0 earns at least **0.05** more than the baseline, with a worst
  case no more than **0.05** below the baseline's.

**Worst case, pessimistically.** When a punisher is nearly indifferent between responses, which
one an exact best-response computation happens to pick can change our payoff substantially (§4.3).
All worst cases in §7 therefore use a *pessimistic* punisher: it maximises its own payoff, but
among responses within 0.01 chips of its best at each decision it picks the one worst for us.

**Noise scale.** Every number is an exact evaluation of the strategy computed, but strategies are
approximate solutions. Re-solving six restricted responses with twice the iterations, or from a
random start, changed gain by at most 0.010 (median 0.001) and worst case by at most 0.030
(median 0.007). The 0.05 tolerances above are therefore conservative.

### 3.5 Methods

- **Linear mixing** (reference only): (1 − w) · CFR + w · BR(model), w = 0, 0.1, …, 1.
- **Three-player restricted Nash response (RNR).** In a modified game each opponent,
  independently, follows its biased model with probability p and is otherwise *free*: it knows
  its mode and plays to maximise its own payoff. We observe neither mode. Solved with CFR+
  (600 iterations; largest convergence gap across all runs 0.0076). At p = 0 both opponents are
  always free, so the modified game *is* the real game and RNR is an independent, shorter
  CFR+ solve of it (600 iterations; players' combined best-response gain 0.0096, against 0.0023
  for the CFR table). Making both free opponents minimise our payoff would make them a coalition; we
  exclude that. Restricting only one opponent (a one-sided variant) was studied and rejected:
  the trusted opponent can punish it for over 3 chips/hand (§7.3).
- **Engine.** A vectorised tree engine for CFR on modified games, best responses and exact
  evaluation, verified against OpenSpiel (§9).
- **Identifiers.** Exact Bayesian posteriors that marginalise over unseen cards: over the 16 type
  pairs at a known alpha (§6.1), or over each opponent's type *and* alpha on a grid (§6.2).
- **Session simulator.** Plays hands between fixed and adaptive players; adaptive players update
  their posterior every hand and their strategy every 25 hands (§8).

---

## 4. Equilibrium is not a floor

**4.1 Extreme opponents beat a non-adapting equilibrium player [EXACT].** Playing CFR against
two over-raisers at alpha = 1.0 loses 0.185 chips/hand relative to equilibrium; against an
over-raiser and an over-caller, 0.692. In two-player zero-sum games, deviations can only help
an equilibrium player. Here the equilibrium carries no floor. This value doubled when the
baseline converged from 1,000 to 5,000 iterations: at extreme bias, play is driven into rarely
visited parts of the tree where equilibria are least pinned down.

**4.2 Opponents' mistakes go to the third player [EXACT].** With one over-raiser in seat 1, an
equilibrium opponent in seat 2, and us playing CFR in seat 0, the raiser loses 0.329 per hand at
alpha 0.35; we gain 0.126 and the other opponent gains 0.203. At alpha 0.1 the split is −0.086,
+0.023, +0.063. At small and moderate bias the third player collects more of the mistake than we
do — a quantitative Morton's theorem. Adapting is how you claim it.

**4.3 An almost-indifferent opponent is a kingmaker [EXACT].** With all three at equilibrium,
an opponent that best-responds to the others gains almost nothing, so it is nearly indifferent
among many responses. Restricted to responses within 0.001 chips of its best at each decision, it
can still move *our* payoff over a wide range: from seat 1 with seat 2 deviating, from −0.23 to
+0.11; from seat 0 with seat 2 deviating, from −0.11 to +0.23. An opponent with nothing to gain
can decide, at almost no cost to itself, which of the other two players pays.

**4.4 Learning opponents reach a second equilibrium [EXACT].** Holding our CFR strategy fixed in
seat 0, the two opponents learn simultaneously, each self-interested, with no collusion. From a
uniform start and two random starts, all three runs converge to the same profile (best-response
gaps below 0.0012, tighter than our CFR baseline's own convergence): **we lose 0.391 while seat 1
gains 0.141 and seat 2 gains 0.251** relative to equilibrium. The equilibrium our CFR table belongs
to is reached from none of them. Both opponents prefer the new equilibrium, and neither needs to
coordinate to reach it.

The effect is not an artifact of how well our CFR table converged. The table is a sound
equilibrium against any *single* opponent changing strategy (nash_conv 0.0023); the new profile
is reached by *both* opponents changing together, which Nash equilibrium does not rule out. And an
independently computed equilibrium-level strategy — the p = 0 restricted response, from different
code and less converged (combined gap 0.0096) — suffers nearly the same fate: after 300
iterations of opponent learning we lose 0.349, with seat 1 gaining 0.124 and seat 2 gaining 0.224
(gaps below 0.005), against 0.384 for the CFR table at the same point. Both strategies come from
the CFR family, so a structurally different equilibrium might fare differently. Note also that the
new profile is stable for the opponents only while our strategy stays fixed; if we keep adapting,
we can recover some of the loss. **A fixed equilibrium strategy is not safe against opponents who
keep learning.**

**4.5 Worst case is sensitive to which equilibrium-level strategy you play [EXACT].** Three
approximate equilibria of the same game — our 5,000-iteration CFR table, the same OpenSpiel run
stopped at 1,000 iterations, and the p = 0 restricted response — earn nearly the same when
everyone plays equilibrium, yet differ markedly in worst case (seat 0, alpha 0.35):

| pair | CFR, 5,000 it. | CFR, 1,000 it. | RNR p = 0 |
|---|---|---|---|
| FO | −0.025 | +0.067 | +0.070 |
| RO | −0.103 | −0.071 | −0.050 |
| RR | −0.194 | −0.174 | −0.132 |

The two CFR checkpoints differ in strategy by only 0.004 (reach-weighted total variation), so
the worst-case measure is highly sensitive to small strategy differences. The p = 0 solve
differs mostly at rarely reached information sets (0.080 there, 0.008 where play usually goes),
which equilibrium conditions barely constrain but a punisher's deviation reaches. In every pair tested, the more converged CFR checkpoint was the more punishable. The pattern does
not extend to restricted responses, whose worst cases moved by at most 0.03, in either direction,
when re-solved with twice the iterations (§9). Why the CFR checkpoints differ we have not
established. We therefore measure adaptation
against the **best of all three** on each axis; the headline in §7 is unchanged whether or not
the 1,000-iteration strategy is included.

**4.6 Over-raisers make equilibrium play punishable [EXACT, exact punisher].** Over the grid of
independent biases (seat 0), the worst case of the better of the two equilibrium-level strategies
(the 5,000-iteration CFR table and the p = 0 restricted response) depends almost entirely on
whether a raiser is at the table. With one deviator at alpha 0.6 and an equilibrium player beside it:

| deviator | in seat 1 | in seat 2 |
|---|---|---|
| over-folder | +0.025 | +0.026 |
| over-raiser | **−0.283** | **−0.298** |
| over-caller | −0.164 | +0.046 |

Next to over-folders, equilibrium play stays safe at every alpha; next to an over-raiser it grows
steadily more punishable with the raiser's alpha, whatever the other opponent does. With two
deviators, the worst case is set by the more dangerous of the two.

---

## 5. The value of exploiting mistakes

### 5.1 Equal bias

Seat 0, alpha 0.35 **[EXACT]**: gain available to a best response, and what CFR collects
without adapting:

| pair | CFR gain | BR gain | | pair | CFR gain | BR gain |
|---|---|---|---|---|---|---|
| FF | +0.171 | +1.846 | | RR | +0.219 | +1.217 |
| RF | +0.207 | +1.415 | | CC | +0.189 | +0.687 |
| FR | +0.326 | +1.379 | | OF | +0.088 | +0.950 |
| CF | +0.199 | +1.191 | | RO | +0.126 | +0.879 |
| CR | +0.296 | +1.087 | | OR | +0.240 | +0.802 |
| RC | +0.208 | +0.997 | | OC | +0.105 | +0.489 |
| FC | +0.183 | +0.972 | | CO | +0.095 | +0.418 |
| FO | +0.077 | +0.776 | | | | |

Best responses are catastrophically exposed: their worst case ranges from −1.1 to −4.0.

**Shape and scale [EXACT].** Best-response gain grows monotonically with alpha, near linearly for
alpha ≤ 0.2. Compared at equal *deviation* rather than equal alpha, folding errors are by far the
most punishable: at 0.10 overridden decisions per hand, one over-folder is worth +0.98 against
+0.10 for an over-raiser and +0.16 for an over-caller. Apparent saturation of over-folder gains at
high alpha is an artifact of alpha running out of room, not of folding errors becoming harmless.

### 5.2 Independent biases [EXACT]

Seat 0, every combination of types with each opponent's alpha in 0, 0.05, …, 0.6 (1,369 distinct
tables; every cell shared with the equal-bias study reproduces it exactly).

**The landscape is smooth.** Predicting each cell from its neighbours on a grid twice as coarse
(steps of 0.1) misses best-response gain by a median of 0.005 (largest 0.040, on values up to 3.1)
and the equilibrium-level worst case by a median of 0.001 (largest 0.057). Coarse grids are enough.

**Two deviators are worth less than their sum.** A best response's gain against both together is a
median 86% of the sum of its gains against each alone (range 59%–111%).

**Types grow differently.** Best-response gain with one deviator beside an equilibrium player
(ranges are over the deviator's two seats):

| deviator | alpha 0.1 | alpha 0.3 | alpha 0.6 |
|---|---|---|---|
| over-folder | +0.14 to +0.26 | +0.62 to +0.82 | +1.34 to +1.49 |
| over-raiser | +0.15 to +0.22 | +0.64 to +0.74 | **+1.77 to +1.80** |
| over-caller | +0.13 to +0.14 | +0.36 to +0.41 | +0.78 to +0.92 |

Over-folders are worth most at moderate alpha; over-raisers accelerate and overtake them. As §3.3
warns, alpha is not comparable across types, so this compares tilts, not equal amounts of deviation.

---

## 6. Identifying opponents [EXACT likelihoods, MC sessions]

### 6.1 Type pairs at a known alpha

An exact Bayesian posterior over the 16 ordered pair hypotheses, marginalising over unknown hole
cards, needs no privileged information. Hands until the posterior on the true pair reaches 0.9
(seat 0, 50 sessions per pair; median over pairs of each pair's median):

| alpha | actions only | + showdown cards | + all cards revealed |
|---|---|---|---|
| 0.10 | 112 | 108 | 54 |
| 0.20 | 46 | 45 | 31 |
| 0.35 | 22 | 19 | 14 |

Revealing cards after every hand helps 1.4–2.1×; showdown cards alone barely help, because the
betting already carries most of the information. At alpha 0.1 some pairs are not identified within
250 hands in up to 60% of sessions. Rare actions move the belief most: where the equilibrium plays
(raise, call, fold) = (0.4, 0.1, 0.5), a call is 4.15× more likely from an over-caller at alpha 0.35,
a raise only 1.53× more likely from an over-raiser.

The deviation most valuable to exploit (folding, §5) is also among the slowest to detect: 45 hands
for FO versus 31 for RO at alpha 0.35, and 7–9 for pairs of two deviators such as RR and RC.

### 6.2 A continuous bias for each opponent [MC, PRELIM: 8 sessions per case]

Assuming one known alpha fails in a specific way. Against a mild over-raiser (alpha 0.1), an
identifier that assumes alpha 0.35 gets the *type* right — across all of Leduc's decision points the
evidence favours "raiser" over "equilibrium player" — but becomes certain the raiser is at 0.35,
over three times the truth, which invites over-exploitation. (A one-decision analysis predicts the
opposite failure, mislabelling the raiser as an equilibrium player; the full game does not.)

We therefore give each opponent its own type and alpha: equilibrium (alpha 0), or R, C or F at
alpha 0.05, 0.10, …, 0.60 — 37 hypotheses per opponent, 1,369 jointly, with 25% prior weight on
equilibrium play. The tilt is linear in alpha, so each possible deal is replayed once and the whole
grid is evaluated at once.

| true opponents | hands to identify the types: grid / fixed-alpha | grid's alpha estimates (200 hands) |
|---|---|---|
| R 0.1, O | 136 / 55 | 0.094, 0.004 |
| F 0.35, R 0.1 | 35 / 35 | 0.364, 0.120 |
| C 0.12, F 0.27 | **45 / 88** (fixed-alpha: never, in 2 of 8) | 0.119, 0.300 |

When the biases differ from what the fixed-alpha identifier assumes, the grid identifier is faster
and steadier, and it recovers each alpha to within about 0.03. It is slower only at telling a mild
bias from none, a distinction worth little to exploit.

## 7. Restricted Nash response

### 7.1 Headline [EXACT, pessimistic punisher]

| | |
|---|---|
| **Safe exploitation** (≥ 0.05 extra profit, worst case within 0.05 of baseline) | **41 of 45** at alpha 0.35; 9 of 12 at alpha 0.2 †; 8 of 15 at alpha 0.1 |
| … median extra profit where it exists | **+0.33** at alpha 0.35 (39% of a best response's extra); +0.16 at 0.2; +0.09 at 0.1 |
| … requiring ≥ 0.10 extra profit | 39 of 45 at alpha 0.35; 2 of 15 at alpha 0.1 |

† alpha 0.2: seat 0 only, 12 of 15 pairs completed, exact rather than pessimistic punisher.

A stricter question — is the restricted response *both* more profitable *and* at least 0.05 safer
than the baseline? — holds in 33 of 45 settings at alpha 0.35 and 1 of 15 at alpha 0.1. We use safe
exploitation as the headline, since earning more without becoming less safe is what a restricted
response is for.

In two-player zero-sum games restricted responses are epsilon-safe best responses, so no mixture of
equilibrium and best response can beat them (Johanson et al., 2007; Johanson & Bowling, 2009). No
such guarantee holds with three players; empirically, the restricted response still beat
per-decision linear mixing at equal profit in 162 of 166 comparisons.

### 7.2 Seat 0, alpha 0.35 [EXACT, pessimistic punisher]

Baseline (best of the three equilibrium-level strategies on each axis) and the most profitable
safe restricted response (gain / worst case):

| pair | baseline | best safe RNR | best response |
|---|---|---|---|
| RR | +0.231 / −0.135 | p=0.75: +0.954 / +0.063 | +1.217 / −1.107 |
| RF | +0.215 / −0.058 | p=0.5: +0.807 / −0.007 | +1.415 / −3.320 |
| RC | +0.217 / −0.050 | p=0.5: +0.675 / +0.079 | +0.997 / −1.200 |
| CC | +0.192 / −0.126 | p=0.9: +0.621 / −0.036 | +0.687 / −1.962 |
| FF | +0.171 / −0.046 | p=0.5: +0.575 / +0.039 | +1.846 / −3.860 |
| CR | +0.306 / −0.135 | p=0.5: +0.653 / +0.043 | +1.087 / −1.163 |
| FR | +0.328 / −0.135 | p=0.5: +0.661 / +0.047 | +1.379 / −2.519 |
| CF | +0.199 / −0.126 | p=0.5: +0.481 / −0.010 | +1.191 / −3.726 |
| FC | +0.185 / +0.035 | p=0.5: +0.439 / +0.063 | +0.972 / −3.969 |
| CO | +0.095 / −0.126 | p=0.9: +0.338 / −0.097 | +0.418 / −2.211 |
| OF | +0.088 / −0.057 | p=0.5: +0.311 / −0.008 | +0.950 / −3.419 |
| RO | +0.129 / −0.050 | p=0.25: +0.350 / −0.063 | +0.879 / −1.178 |
| OC | +0.106 / +0.035 | p=0.5: +0.293 / +0.042 | +0.489 / −2.319 |
| OR | +0.242 / −0.135 | p=0.5: +0.375 / +0.124 | +0.802 / −1.709 |
| FO | +0.077 / −0.024 | p=0.25: +0.141 / −0.047 | +0.776 / −3.104 |

Typically the safe restricted response two- to four-folds the baseline's profit while its worst
case stays close to the baseline's, often above it; a best response's worst case is −1.1 to −4.0.

### 7.3 The safe region, and its abrupt end [EXACT]

FF, seat 0, fine grid in p:

| p | 0 | 0.25 | 0.5 | 0.6 | 0.65 | 0.7 | 0.75 | 0.9 | 0.97 |
|---|---|---|---|---|---|---|---|---|---|
| gain | +0.170 | +0.299 | +0.575 | +0.974 | +1.323 | +1.439 | +1.580 | +1.708 | +1.796 |
| worst | −0.056 | **+0.123** | +0.039 | −0.339 | −0.521 | −0.572 | −0.528 | −0.531 | −1.309 |

The computed solutions show two regimes: a *safe* regime up to p = 0.5 where the worst case
stays above equilibrium, and an *exploit* regime from p = 0.65 where exposure plateaus near −0.53
while gain keeps rising, then a final cliff near the pure best response. The switch is abrupt in
the solutions we computed. Because the modified game is general-sum and CFR+ carries no
uniqueness guarantee there, we cannot rule out that the jump reflects the solver moving between
different solutions of the modified game rather than a property of restricted responses as such.
RO shows the same shape
from seat 2, with the switch between p = 0.25 (+0.588 / +0.141, against CFR's +0.212 / −0.088)
and p = 0.5 (+0.763 / −0.289).

**One-sided restriction fails.** Trusting one opponent's model completely gives gains of
+1.28 to +1.69, but that opponent can then punish for −3.2 to −3.6 — a failure that only shows
if the worst case is checked against the trusted opponent too. The symmetric version at p = 0.9 achieves a similar gain (+1.708) with a
worst case of −0.531.

### 7.4 By seat [EXACT, pessimistic punisher]

Safe exploitation at alpha 0.35:

| seat | settings | median extra profit |
|---|---|---|
| 0 | 15 of 15 | +0.282 |
| 1 | 14 of 15 | +0.324 |
| 2 | 12 of 15 | +0.401 |

There is no clear seat ordering in how often safe exploitation exists once each seat is measured
against its own equilibrium value. The strongest single point: RR from seat 2 at p = 0.5, +1.016
gain with a worst case of +0.453 to +0.475 across three independent solves.

### 7.5 Weaker bias [EXACT]

Safe exploitation exists in 9 of 12 completed pairs at alpha 0.2 (median extra +0.16) and 8 of 15
at alpha 0.1 (median extra +0.09; only 2 offer +0.10 or more). FF, seat 0, alpha 0.1 (pessimistic
punisher):

| | CFR | p=0.25 | p=0.5 | p=0.75 | p=0.9 | BR |
|---|---|---|---|---|---|---|
| gain | +0.049 | +0.057 | +0.083 | +0.143 | +0.314 | +0.379 |
| worst | -0.066 | -0.034 | +0.036 | +0.001 | -0.527 | -3.536 |

Mild deviators are both slower to identify (§6) and less valuable to exploit safely.

### 7.6 Which opponent is dangerous to trust [EXACT]

Split by which opponent switches to punishing (seat 0, alpha 0.35, relative to equilibrium):

| pair | p | seat 1 punishes | seat 2 punishes |
|---|---|---|---|
| OF | 0.9 | +0.547 | **−0.834** (the folder) |
| RO | 0.9 | **−0.498** (the raiser) | +0.389 |
| OR | 0.9 | +0.172 | **−0.709** (the raiser) |

Swapping the raiser's seat (RO vs OR) moves the danger with him. Across all seats and alphas,
when one opponent deviates and the other plays equilibrium, **the deviator is the harder punisher
in 56 of 68 cases**, and in 49 of the 54 cases where the two punishers differ by at least 0.10.
A natural interpretation — our commitment to their pattern is what they can turn against us — is
a hypothesis we have not tested directly. Exceptions are mostly over-caller pairs, where commitment is smallest (e.g. OC at p = 0.9: the
equilibrium opponent punishes for −0.040, the over-caller not at all, +0.224). With two deviators there is no simple rule.

### 7.7 Robustness to model errors [EXACT]

Strategies tuned for FF at alpha 0.35 (seat 0), played against other opponents (gain):

| | FF 0.1 | FF 0.2 | FF 0.35 | FF 0.5 | RR | CC | OO |
|---|---|---|---|---|---|---|---|
| CFR | +0.049 | +0.098 | +0.171 | +0.242 | +0.219 | +0.189 | 0.000 |
| RNR 0.25 | +0.082 | +0.173 | +0.299 | +0.410 | +0.179 | +0.209 | −0.015 |
| RNR 0.5 | +0.131 | +0.319 | +0.575 | +0.797 | +0.187 | +0.117 | −0.072 |
| RNR 0.75 | +0.118 | +0.749 | +1.580 | +2.258 | −0.069 | −0.313 | −0.569 |
| RNR 0.9 | +0.017 | +0.750 | +1.708 | +2.475 | −0.192 | −0.530 | −0.785 |
| BR | −0.027 | +0.789 | +1.846 | +2.677 | −0.933 | −0.961 | −0.920 |

At p ≤ 0.5 the strategy is robust: the right type with the wrong alpha still beats CFR, and the
wrong type costs at most 0.07. At p ≥ 0.75 the wrong type costs 0.3–0.8. For this pair, the
regime switch of §7.3 is also where robustness to modelling error collapses. That suggests a rule
for combining identification with adaptation — stay at low p until the posterior is confident —
but it rests on one pair and one seat, and §8 shows that a confident posterior can still be wrong.

### 7.8 Where safe exploitation is not found

At alpha 0.35: FO, OC and OF from seat 2, and RO from seat 1. At alpha 0.2: OR, RO, FO (seat 0).
At alpha 0.1: CF, CO, FO, OF, OR, RO, RR. Nearly all are single-deviator pairs, where there is less
to gain and the baseline already offers comparable protection. Several were tested at only a few
values of p; where a safe region exists it usually lies at p ≤ 0.5 and can end below p = 0.5.

### 7.9 When both opponents adapt [EXACT]

If both opponents learn best responses to our fixed strategy and to each other (500 iterations,
gaps below 0.0012), exposure is far larger than against a single punisher (FF, seat 0):

| | single punisher (pessimistic) | both adapt |
|---|---|---|
| CFR | −0.061 | −0.391 |
| RNR p = 0.5 | +0.039 | −0.213 |
| RNR p = 0.9 | −0.531 | −2.246 |
| BR | −3.860 | −4.289 ‡ |

‡ 120 iterations, gaps up to 0.011.

RNR at p = 0.5 is hurt roughly half as much as CFR. These are static stress tests, in which we
never adapt back; §8 plays the same situation out with every player adapting.

### 7.10 Independent biases, and a finer search over trust [EXACT; exact punisher — pessimistic re-evaluation and stability re-solves pending]

Opponent pairs with an over-raiser (RR, FR, RF, CR, RC), all three seats, each opponent's alpha in
0, 0.2, 0.4, 0.6 independently: 63 settings per seat. (Steps of 0.2 suffice: predicting the
0.1-step cells of §5.2's grid from them misses the equilibrium-level worst case by a median of
0.003, at most 0.057.) Each
setting is solved at p = 0.25 and 0.5, then **searched adaptively**: bisect between the largest safe
p and the smallest unsafe p above it until they are within 0.07, with at most three extra solves.

| we sit in | safe exploitation | median extra profit | largest | share of the best response's extra |
|---|---|---|---|---|
| seat 0 | **57 of 63** | **+0.70** | +1.86 | 67% |
| seat 1 | **62 of 63** | **+0.67** | +2.16 | 64% |
| seat 2 | **58 of 63** | **+0.89** | +2.34 | 65% |

Safe extra profit grows with both opponents' biases, reaching +1.3 to +2.3 when both are at 0.6.

**This is roughly twice the main study's figures**, and the reason is the search, not the opponents:
the safe region typically ends at p = 0.6–0.8 (median largest safe p 0.70, 0.70 and 0.60 by seat),
between the points p = 0.5 and 0.9 that the main study mostly tested. The main study's headline
(median extra profit 0.33, 39% of the best response's extra) is therefore likely conservative; the
same search on its settings is running.

**The result does not hinge on the tolerance.** The best safe points do not sit at its edge: their
worst case is typically *above* the best equilibrium-level strategy's (median +0.036; at or above it
in 67% of settings). Tightening "no material loss" from 0.05 to 0.03, 0.02 and 0.01 leaves 173, 172
and 170 of 189 settings; requiring no loss at all leaves **166 of 189**, median extra profit +0.74.

**Lone deviators are the hard case.** Median largest safe p:

| table | largest safe p |
|---|---|
| lone over-caller beside an equilibrium player | 0.90 |
| two deviators | 0.70 |
| lone over-raiser beside an equilibrium player | 0.50 |
| lone over-folder beside an equilibrium player | 0.34 |

With two deviators there is a cushion: while one punishes, the other keeps playing its exploitable
style. A lone over-folder or over-raiser offers none. Position also matters for lone deviators, but
with no consistent rule across seats at these sample sizes.

**Who is dangerous to trust: the opponent you exploit most.** At p = 0.5, the deviator is the harder
punisher in 47 of 54 single-deviator settings, and the *more biased* opponent in 77 of 90 settings
with two deviators at different biases. Type matters less: a raiser facing a folder or caller is the
harder punisher in only 66 of 108. §7.6's finding generalises, with degree of bias as the driver.

**Where safe exploitation fails** (12 settings), it does so in one of two ways. Either the safe
region ends before the extra profit reaches 0.05 — seat 2 facing a lone over-folder directly before
it stays safe only to p ≈ 0.13, earning +0.046 — or even the smallest p tried is unsafe. The second
kind may partly reflect equilibrium selection rather than a cost of exploiting: at very small p a
restricted response is just another equilibrium-level strategy, and those differ in worst case by up
to 0.09 on their own (§4.5).

## 8. Identification and adaptation in play [EXACT expected payoffs; PRELIM: single sessions]

A session simulator seats three players, each fixed or adaptive. An adaptive player runs the
identifier of §6.2 on the other two (with a 1% allowance for random play, so one surprising action
cannot rule a hypothesis out forever) and every 25 hands plays

    (1 − w) · base + w · BR(posterior prediction of the opponents),   w = c^beta,

where c rescales the probability of the most likely type pair to run from 0 at the prior to 1 when
certain, and beta = 1 here. The prediction mixes each opponent's hypotheses, weighted by posterior
and by how likely each is to reach each decision; it is exact (expected payoff against it equals
the posterior-weighted payoff, to 1e−17). At every update we record each player's *exact* expected
payoff for the current profile: realised chips are far too noisy (one adaptive player earned
+0.71 ± 0.37 chips/hand over 300 hands against an exact +1.01).

**When the model is right, adaptation works.** Seat 0 adapts against an over-folder (0.35) and an
over-raiser (0.2). Within 50 hands it is certain of their types and estimates their alphas at about
0.40 and 0.18; its expected payoff rises from +0.21 to **+1.01** relative to equilibrium and stays
there, both opponents now paying. Its strategy settles: changes per update fall from 0.24 to under
0.01 (reach-weighted total variation).

**When all three adapt, nobody's model fits.** Three adaptive players start as an over-folder, an
over-raiser and an over-caller (alpha 0.3 each). Expected payoffs swing by up to 0.6 chips/hand in
the first 100 hands as each responds to the others' responses, then drift into a lopsided
arrangement (after 250 hands: +0.13, +0.19, −0.32), not toward equilibrium. **All three players
reach full confidence within 25–50 hands, though all three models are wrong** — no one is a fixed
tilt of equilibrium once everyone adapts. Both opponents confidently label the original over-folder
an over-*caller*, which describes how its responses now play, not what it was. Once posteriors lock,
many updates change nothing.

Confidence in this sense measures certainty *within* the hypothesis family, not whether the family
fits, so a schedule that adapts harder as confidence grows gives no protection against opponents
outside the model — and adaptive opponents always are. A check of how well the most likely
hypothesis predicts the observed actions, or restricted responses chosen from a solved bank (safe
by construction), are natural remedies; the bank is being filled from the grid of §5.2.

## 9. Validity checks

| check | result |
|---|---|
| Engine vs OpenSpiel | identical to 6 decimals for expected values and best responses, including a best response against a biased, non-equilibrium profile |
| All results on the same baseline | saved strategies from every stage re-evaluate with 0.0 difference on the 5k baseline |
| p = 0 reuse | cached and from-scratch results identical to every digit |
| **Solution stability** | six settings re-solved with 1,200 iterations and from a random start: gain changes ≤ 0.010, worst case ≤ 0.030 |
| **Pessimistic punisher** | applied to all 64 settings; median change in any worst case 0.000, but up to −0.24 in some, so it is the default |
| **Tie-breaking, all at equilibrium** | large: our payoff can range over ±0.23 (§4.3); single-value claims withdrawn |
| **Second equilibrium** | three starting points converge to the same profile, gaps < 0.0012; an independently computed equilibrium-level strategy (RNR at p = 0) loses 0.349 under the same test |
| Baseline choice | safe exploitation measured against the best of three equilibrium-level strategies |
| Deviator claim | holds with a 0.10 materiality threshold (49 of 54) |
| OO control | no gain to find, as expected |
| Independent-bias grid | 1,369 tables; every cell shared with the equal-bias study reproduces it exactly |
| Continuous-bias identifier | tilt formula equals the opponent construction (2e−16); agrees with the 16-pair identifier on shared hypotheses (2e−14); matches brute force over every deal from seat 1 (3e−15) |
| Simulator's prediction | expected payoff against the predicted opponent equals the posterior-weighted payoff (4e−17) |
| Grid restricted responses (§7.10) | all 189 settings complete; each seat's equilibrium value recovered from the grid matches the engine; verdicts stable as the tolerance tightens from 0.05 to 0.01 (177 → 170 settings) |
| Pessimistic re-scoring tool | reproduces the main study's pessimistic evaluation exactly on the same strategies (0 difference) |

## 10. Discussion

**Safety needs a new reference point.** Two-player safe exploitation measures risk against the
game value. In three-player games there is no such value, a single equilibrium is not a floor,
and learning opponents can move to an equilibrium that is worse for you. Our results suggest a
practical substitute: measure risk against the best available equilibrium, under a
self-interested — not spiteful — punisher.

**How you adapt matters.** A restricted response that trusts neither opponent fully finds a
safe region, usually at low trust, where profit rises several-fold without a material loss of
robustness; beyond it, trust buys profit only at a sharp cost in safety. That region reaches further
than a coarse search reveals — typically to p = 0.6–0.8 — and finding its edge roughly doubles the
safe profit (§7.10).

**Confidence is not correctness.** A posterior can be certain and wrong when no hypothesis fits,
and among adaptive players that is the normal case (§8). Adaptation rules keyed to confidence need
a separate check that the model fits at all.

**Compare safety carefully.** Profit is a smooth function of strategy; worst-case safety is not.
Two strategies that earn the same can differ by 0.1 chips/hand in how punishable they are, and
when every player is at equilibrium the punisher's choice among near-equal responses matters as
much as its best response. Safety comparisons in multiplayer games need a stated noise scale.

**Most of the risk comes from the opponent you are exploiting.** A policy committed to an
opponent's mistake is exposed to that opponent's correction. This suggests restricting more
tightly against the deviator than against the other player.

---

## 11. Limitations and open items

1. **Opponent class:** single-action tilts. Independent biases are mapped for seat 0 (best-response
   gain and equilibrium safety, §4.6, §5.2), and restricted responses over them for pairs with an
   over-raiser in all seats (§7.10); pairs without one (FF, FC, CF, CC) are not yet solved.
2. **Persistent types:** RNR redraws opponent modes every hand; real opponents keep their style.
3. **Adaptation in play is preliminary** (§8): single sessions, best-response adaptation only (the
   bank of restricted responses is incomplete), and all players update on the same hand.
4. **Per-opponent restriction** is untested; §7.6 suggests restricting more tightly against the
   deviator.
5. **Choice of equilibrium-level baseline.** Worst cases differ across equilibrium-level strategies
   (§4.5). A refinement that pins down rarely reached play (e.g. a trembling-hand style
   perturbation) is the principled fix; exact multiplayer equilibrium solvers could
   enumerate alternatives.
6. **alpha 0.2** covers seat 0 and 12 of 15 pairs, with the exact rather than pessimistic
   punisher; independent-bias results (§4.6, §5.2, §7.10) also use the exact punisher. Pessimistic
   re-evaluation of §7.10 is running, as are re-solves of 16 of its best safe points (different
   random start; half also at doubled iterations) — those points lie near the abrupt switch, where
   solutions may be least stable.
7. **Mechanisms untested:** the explanation offered for §7.6 (commitment to a deviator's pattern)
   and the "cushion" explanation for lone deviators (§7.10) are hypotheses; §4.5's CFR-checkpoint
   difference is unexplained.
8. **The main study's search over trust was coarse** (mostly p = 0, 0.5, 0.9); §7.10 suggests its
   headline understates safe profit. The adaptive search on its settings is running.

## 12. Conclusion

In three-player Leduc Hold'em, equilibrium play offers no floor: extreme opponents beat it, an
almost-indifferent opponent can decide who pays, independently learning opponents can settle into
an equilibrium that costs a fixed equilibrium player 0.39 chips/hand, and an over-raiser at the
table makes equilibrium play punishable. Yet safe exploitation is possible: a symmetric
three-player restricted Nash response earns substantially more than any equilibrium-level strategy
we tested without a material loss of worst-case safety in 41 of 45 settings at moderate bias — and,
searching more finely over trust, in 177 of 189 settings over a grid of independent biases — up to
an abrupt switch beyond which trust costs safety sharply. Worst-case safety is far more sensitive
than profit to small changes in strategy, a fact any comparison of multiplayer strategies has to
account for. And when every player adapts, posteriors grow confident in models that no longer fit
— so in multiplayer games, knowing when your model is wrong may matter as much as fitting it.

---

## References

**✓** details checked during drafting; **⚠** details to confirm. All references are expected to have at least 3
citations; those marked **[confirm ≥ 3 citations]** should be checked on Google Scholar before submission.

- ✓ Abou Risk, N. & Szafron, D. (2010). Using counterfactual regret minimization to create competitive multiplayer poker agents. *AAMAS*, 159–166.
- ✓ Albrecht, S. V., Crandall, J. W. & Ramamoorthy, S. (2016). Belief and truth in hypothesised behaviours. *Artificial Intelligence* 235, 63–94.
- ✓ Albrecht, S. V. & Stone, P. (2017). Reasoning about hypothetical agent behaviours and their parameters. *AAMAS*, 547–555. **[confirm ≥ 3 citations]**
- ✓ Albrecht, S. V. & Stone, P. (2018). Autonomous agents modelling other agents: A comprehensive survey and open problems. *Artificial Intelligence* 258, 66–95.
- ✓ Brown, N. & Sandholm, T. (2019). Superhuman AI for multiplayer poker. *Science* 365(6456), 885–890.
- ✓ Ganzfried, S. (2023). Safe equilibrium. *IEEE CDC*, 5230–5236. arXiv:2201.04266. **[confirm ≥ 3 citations]**
- ✓ Ganzfried, S. & Sandholm, T. (2015). Safe opponent exploitation. *ACM Transactions on Economics and Computation* 3(2), 8:1–8:28.
- ✓ Ganzfried, S., Wang, K. A. & Chiswick, M. (2024). Opponent modeling in multiplayer imperfect-information games. *DAI '24*, 39–45. **[confirm ≥ 3 citations]**
- ✓ Hoehn, B., Southey, F., Holte, R. C. & Bulitko, V. (2005). Effective short-term opponent exploitation in simplified poker. *AAAI*, 783–788.
- ✓ Johanson, M. & Bowling, M. (2009). Data biased robust counter strategies. *AISTATS*, 264–271.
- ✓ Johanson, M., Zinkevich, M. & Bowling, M. (2007). Computing robust counter-strategies. *NIPS 20*, 721–728.
- ⚠ Lanctot, M. et al. (2019). OpenSpiel: A framework for reinforcement learning in games. arXiv:1908.09453.
- ✓ McCracken, P. & Bowling, M. (2004). Safe strategies for agent modelling in games. *AAAI Fall Symposium on Artificial Multi-agent Learning*.
- ✓ Southey, F., Bowling, M., Larson, B., Piccione, C., Burch, N., Billings, D. & Rayner, C. (2005). Bayes' bluff: Opponent modelling in poker. *UAI*, 550–558.
- ⚠ Southey, F., Hoehn, B. & Holte, R. C. (2009). Effective short-term opponent exploitation in simplified poker. *Machine Learning*. doi:10.1007/s10994-008-5091-5 (volume and pages unconfirmed). **[confirm ≥ 3 citations]**
- ✓ Szafron, D., Gibson, R. & Sturtevant, N. (2013). A parameterized family of equilibrium profiles for three-player Kuhn poker. *AAMAS* (pages unconfirmed). **[confirm ≥ 3 citations]**
- ⚠ Tammelin, O. (2014). Solving large imperfect information games using CFR+. arXiv:1407.5042.
- ✓ Zinkevich, M., Johanson, M., Bowling, M. & Piccione, C. (2008). Regret minimization in games with incomplete information. *NIPS 20*, 1729–1736.

---

## Appendix: reproduction

`rnr_package/`: `tree_engine.py` (build and verify the engine), `rnr_general.py` (restricted
responses and linear baselines, with an optional separate alpha per opponent), `overnight.py` and
`overnight_grid.py` (all jobs, including the independent-bias grid with adaptive search over p),
`analyze_het.py`, `reeval_pessimistic.py`, `gaps_check.py` (misspecification and both-adapt),
`q2_identify.py` and `q2_grid_identify.py` (identifiers), `session_sim.py` (play), and
`audit_identifier.py`, `audit_solver.py` (independent checks), `reeval_het.py` (pessimistic
re-scoring of the grid), `stability_het.py` (re-solves of best safe points) and
`overnight_main_adaptive.py` (adaptive search over p for the main study). `cfr_path.py` fixes which
CFR table every script reads. Results: 241 restricted-response solves and 64 linear baselines
(`results/`); in `results_het/`, 3,460 independent-bias baseline runs and 921 restricted-response
solves; all on the 5,000-iteration CFR+ solution. Figures for §7.10: `het_safe_extra.png`,
`het_largest_safe_p.png`, `het_harder_punisher.png`.
