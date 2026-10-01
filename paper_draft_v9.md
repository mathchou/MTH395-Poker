# Safe Exploitation Without a Game Value: Restricted Responses in Three-Player Leduc Hold'em

**Draft v9.** Supersedes v8. All sections now on the 5,000-iteration baseline; worst cases use a pessimistic punisher throughout; solution stability and the second equilibrium are confirmed by converged runs.

Tags: **[EXACT]** exact game-tree evaluation on the 5,000-iteration baseline.
**[MC]** Monte Carlo.
All payoffs are chips per hand, measured relative to the seat's own equilibrium value unless
stated otherwise.

---

## Abstract

In two-player zero-sum games, safe opponent exploitation is well understood: restricted Nash
responses trace the frontier between profiting from an opponent's mistakes and becoming
exploitable, and safety is measured against the game's value. Neither tool carries over to three
or more players, where there is no unique game value and an equilibrium strategy guarantees
nothing. We study the problem exactly in three-player Leduc Hold'em. Against opponents built by
biasing a converged equilibrium toward raising, calling or folding, we find: (i) equilibrium play
is not a floor — a fixed equilibrium strategy loses up to 0.69 chips/hand to extreme opponents,
an almost-indifferent opponent can shift up to 0.23 chips/hand between the other two players, and
two self-interested, non-colluding opponents who keep learning converge, from every start we
tried, to an equilibrium that costs a fixed equilibrium player 0.39 chips/hand; (ii) a
three-player restricted Nash response that trusts neither opponent's model achieves **safe
exploitation** — at least 0.05 chips/hand more profit with no material loss of worst-case safety
— in 41 of 45 settings at moderate bias (median extra profit 0.33), 9 of 12 at intermediate bias
and 8 of 15 at mild bias; (iii) it dominates naive mixing of equilibrium and best response in 162
of 166 comparisons at equal profit; and (iv) worst-case safety is far more sensitive than profit to
small strategy differences and to how a near-indifferent punisher breaks ties, which shapes how
any such comparison must be made.

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
   an exact evaluation across 15 ordered opponent pairs, three seats and two bias levels.
3. **A sensitivity analysis of the worst-case measure.** Worst-case safety moves by up to 0.1
   chips/hand under strategy changes that leave profit unchanged; we calibrate comparisons to
   this noise and to punisher tie-breaking.
4. **Structure of the safe region**: where it lies, which opponent is dangerous to trust, and how
   robust it is to modelling errors.

---

## 2. Related work

**Robust and safe exploitation.** Restricted Nash responses pin the opponent to a model with
probability p and otherwise let it best-respond, tracing a Pareto-optimal gain/exploitability
frontier (Johanson et al., 2007); data-biased responses weight the pinning by data confidence
(Johanson & Bowling, 2009). Epsilon-safe strategies (McCracken & Bowling, 2004) and safe
opponent exploitation (Ganzfried & Sandholm, 2015) bound risk relative to the game value.
Li & Huang (2026) schedule the restriction from confidence sequences with certified
exploitability in two-player Leduc. All are two-player zero-sum. Ganzfried's safe equilibrium
(2023) generalises restricted responses to multiplayer and non-zero-sum games and is the closest
theoretical relative of our construction.

**Opponent modelling.** Leduc Hold'em was introduced with Bayesian opponent modelling (Southey et
al., 2005); short-horizon exploitation in Kuhn poker was studied by Hoehn et al. (2005) and
Southey, Hoehn & Holte (2009). Our identifier is Bayesian reasoning over hypothesised types
(Albrecht, Crandall & Ramamoorthy, 2016), including parameterised types (Albrecht & Stone, 2017;
survey in Albrecht & Stone, 2018). The only multiplayer precedent we found is Ganzfried, Wang &
Chiswick (2024), who model opponents in three-player Kuhn poker. Showdown data are missing not at
random (Guo, 2026); our likelihood models fold decisions and so handles this in principle.

**Multiplayer equilibrium.** CFR (Zinkevich et al., 2008) has no Nash guarantee beyond two
players, though it has produced strong three-player (Abou Risk & Szafron, 2010) and superhuman
six-player agents (Brown & Sandholm, 2019). Three-player Kuhn poker already has a parameterised
family of equilibria (Szafron, Gibson & Sturtevant, 2013), which foreshadows the equilibrium
selection effects we measure. Exact multiplayer methods now exist for small games (Ganzfried,
2026). The observation that an opponent's mistake can benefit a third player is known to poker
players as Morton's theorem or implicit collusion; we quantify it against an equilibrium baseline.

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

- **Linear mixing:** (1 − w) · CFR + w · BR(model), w = 0, 0.1, …, 1.
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
  evaluation, verified against OpenSpiel (§8).

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
can decide, at almost no cost to itself, which of the other two players pays. (An earlier draft
reported single values for these effects; they reflected tie-breaking in the best-response
computation, not a property of the game.)

**4.4 Learning opponents reach a second equilibrium [EXACT].** Holding our CFR strategy fixed in
seat 0, the two opponents learn simultaneously, each self-interested, with no collusion. From a
uniform start and two random starts, all three runs converge to the same profile (best-response
gaps below 0.0012, tighter than our CFR baseline's own convergence): **we lose 0.391 while seat 1
gains 0.141 and seat 2 gains 0.251** relative to equilibrium. The equilibrium our CFR table belongs
to is reached from none of them. Both opponents prefer the new equilibrium, and neither needs to
coordinate to reach it. **A fixed equilibrium strategy is not safe against opponents who keep
learning.**

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
when re-solved with twice the iterations (§8). Why the CFR checkpoints differ we have not
established. We therefore measure adaptation
against the **best of all three** on each axis; the headline in §7 is unchanged whether or not
the 1,000-iteration strategy is included.

---

## 5. The value of exploiting mistakes

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

---

## 6. Identifying opponents [EXACT likelihoods, MC sessions]

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
a raise only 1.53× more likely from an over-raiser. Classifying a single opponent is 2–4× faster
than pinning the ordered pair **[1,000-iteration baseline; not rerun]**.

The deviation most valuable to exploit (folding, §5) is also among the slowest to detect: 45 hands
for FO versus 31 for RO at alpha 0.35, and 7–9 for pairs of two deviators such as RR and RC.

## 7. Restricted Nash response

### 7.1 Headline [EXACT, pessimistic punisher]

| | |
|---|---|
| **Safe exploitation** (≥ 0.05 extra profit, worst case within 0.05 of baseline) | **41 of 45** at alpha 0.35; 9 of 12 at alpha 0.2 †; 8 of 15 at alpha 0.1 |
| … median extra profit where it exists | **+0.33** at alpha 0.35 (39% of a best response's extra); +0.16 at 0.2; +0.09 at 0.1 |
| … requiring ≥ 0.10 extra profit | 39 of 45 at alpha 0.35; 2 of 15 at alpha 0.1 |
| RNR vs linear mixing at equal gain | better in **162 of 166** points; median **+0.69** of worst case |

† alpha 0.2: seat 0 only, 12 of 15 pairs completed, exact rather than pessimistic punisher.

A stricter question — is the restricted response *both* more profitable *and* at least 0.05 safer
than the baseline? — holds in 33 of 45 settings at alpha 0.35 and 1 of 15 at alpha 0.1. We use safe
exploitation as the headline, since earning more without becoming less safe is what a restricted
response is for.

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
+1.28 to +1.69, but that opponent can then punish for −3.2 to −3.6. Draft v5 of this work
reported the one-sided variant as a success because the worst case was only checked against the
untrusted opponent. The symmetric version at p = 0.9 achieves a similar gain (+1.708) with a
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
but it rests on one pair and one seat and should be tested more broadly before being stated as
general.

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

RNR at p = 0.5 is hurt roughly half as much as CFR. These are static stress tests: in practice we
would observe the opponents' change and adapt back, so the relevant cost is exposure multiplied by
the time to detect the change (§6).

## 8. Validity checks

| check | result |
|---|---|
| Engine vs OpenSpiel | identical to 6 decimals for expected values and best responses, including a best response against a biased, non-equilibrium profile |
| All results on the same baseline | saved strategies from every stage re-evaluate with 0.0 difference on the 5k baseline |
| p = 0 reuse | cached and from-scratch results identical to every digit |
| **Solution stability** | six settings re-solved with 1,200 iterations and from a random start: gain changes ≤ 0.010, worst case ≤ 0.030 |
| **Pessimistic punisher** | applied to all 64 settings; median change in any worst case 0.000, but up to −0.24 in some, so it is the default |
| **Tie-breaking, all at equilibrium** | large: our payoff can range over ±0.23 (§4.3); single-value claims withdrawn |
| **Second equilibrium** | three starting points converge to the same profile, gaps < 0.0012 |
| Baseline choice | safe exploitation measured against the best of three equilibrium-level strategies |
| Deviator claim | holds with a 0.10 materiality threshold (49 of 54) |
| OO control | no gain to find, as expected |

## 9. Discussion

**Safety needs a new reference point.** Two-player safe exploitation measures risk against the
game value. In three-player games there is no such value, a single equilibrium is not a floor,
and learning opponents can move to an equilibrium that is worse for you. Our results suggest a
practical substitute: measure risk against the best available equilibrium, under a
self-interested — not spiteful — punisher.

**How you adapt matters more than how fast.** Mixing equilibrium with a best response is
dominated at nearly every point. A restricted response that trusts neither opponent fully finds a
safe region, usually at low trust, where profit rises several-fold without a material loss of
robustness.

**Compare safety carefully.** Profit is a smooth function of strategy; worst-case safety is not.
Two strategies that earn the same can differ by 0.1 chips/hand in how punishable they are, and
when every player is at equilibrium the punisher's choice among near-equal responses matters as
much as its best response. Safety comparisons in multiplayer games need a stated noise scale.

**Most of the risk comes from the opponent you are exploiting.** A policy committed to an
opponent's mistake is exposed to that opponent's correction. This suggests restricting more
tightly against the deviator than against the other player.

---

## 10. Limitations and open items

1. **Opponent class:** single-action tilts, equal alpha for both opponents. Heterogeneous pairs and
   richer styles are untested.
2. **Persistent types:** RNR redraws opponent modes every hand; real opponents keep their style.
3. **Identification and adaptation are not yet combined:** choosing p from the posterior is the
   natural next experiment, guided by §7.7.
4. **Per-opponent restriction** is untested; §7.6 suggests restricting more tightly against the
   deviator.
5. **Choice of equilibrium-level baseline.** Worst cases differ across equilibrium-level strategies
   (§4.5). A refinement that pins down rarely reached play (e.g. a trembling-hand style
   perturbation) is the principled fix; exact multiplayer solvers (Ganzfried, 2026) could
   enumerate alternatives.
6. **alpha 0.2** covers seat 0 and 12 of 15 pairs, with the exact rather than pessimistic
   punisher; alpha grid otherwise 0.1 and 0.35.
7. **Mechanisms untested:** the explanation offered for §7.6 (commitment to a deviator's pattern)
   is a hypothesis; §4.5's CFR-checkpoint difference is unexplained.
8. **Single-opponent identification speed** (§6) is from the 1,000-iteration baseline.

## 11. Conclusion

In three-player Leduc Hold'em, equilibrium play offers no floor: extreme opponents beat it, an
almost-indifferent opponent can decide who pays, and independently learning opponents can settle
into an equilibrium that costs a fixed equilibrium player 0.39 chips/hand. Yet safe exploitation
is possible: a symmetric three-player restricted Nash response earns substantially more than any
equilibrium-level strategy we tested without a material loss of worst-case safety in 41 of 45
settings at moderate bias, and dominates naive mixing almost everywhere. Worst-case safety is far
more sensitive than profit to small changes in strategy, a fact any comparison of multiplayer
strategies has to account for.

---

## References

**✓** checked during drafting; **⚠** confirm before submission.

- ✓ Abou Risk, N. & Szafron, D. (2010). Using counterfactual regret minimization to create competitive multiplayer poker agents. *AAMAS*, 159–166.
- ✓ Albrecht, S. V., Crandall, J. W. & Ramamoorthy, S. (2016). Belief and truth in hypothesised behaviours. *Artificial Intelligence* 235, 63–94.
- ✓ Albrecht, S. V. & Stone, P. (2017). Reasoning about hypothetical agent behaviours and their parameters. *AAMAS*, 547–555.
- ✓ Albrecht, S. V. & Stone, P. (2018). Autonomous agents modelling other agents: A comprehensive survey and open problems. *Artificial Intelligence* 258, 66–95.
- ✓ Brown, N. & Sandholm, T. (2019). Superhuman AI for multiplayer poker. *Science* 365(6456), 885–890.
- ✓ Ganzfried, S. (2023). Safe equilibrium. *IEEE CDC*, 5230–5236. arXiv:2201.04266.
- ✓ Ganzfried, S. (2026). Quadratic programming approach for Nash equilibrium computation in multiplayer imperfect-information games. *Games* 17(1), 9.
- ✓ Ganzfried, S. & Sandholm, T. (2015). Safe opponent exploitation. *ACM Transactions on Economics and Computation* 3(2), 8:1–8:28.
- ✓ Ganzfried, S., Wang, K. A. & Chiswick, M. (2024). Opponent modeling in multiplayer imperfect-information games. *DAI '24*, 39–45.
- ✓ Guo, J. (2026). Safe observation capacity for opponent exploitation under showdown censoring. arXiv:2608.09954.
- ✓ Hoehn, B., Southey, F., Holte, R. C. & Bulitko, V. (2005). Effective short-term opponent exploitation in simplified poker. *AAAI*, 783–788.
- ✓ Johanson, M. & Bowling, M. (2009). Data biased robust counter strategies. *AISTATS*, 264–271.
- ✓ Johanson, M., Zinkevich, M. & Bowling, M. (2007). Computing robust counter-strategies. *NIPS 20*, 721–728.
- ⚠ Lanctot, M. et al. (2019). OpenSpiel: A framework for reinforcement learning in games. arXiv:1908.09453.
- ✓ Li, B. & Huang, L. (2026). Agents that certify their own exploits: Confidence-scheduled restricted responses for safe opponent exploitation. arXiv:2607.28520.
- ✓ McCracken, P. & Bowling, M. (2004). Safe strategies for agent modelling in games. *AAAI Fall Symposium on Artificial Multi-agent Learning*.
- ⚠ Morton, A. (1999). Post to rec.gambling.poker (Morton's theorem). Find a citable published treatment.
- ✓ Southey, F., Bowling, M., Larson, B., Piccione, C., Burch, N., Billings, D. & Rayner, C. (2005). Bayes' bluff: Opponent modelling in poker. *UAI*, 550–558.
- ⚠ Southey, F., Hoehn, B. & Holte, R. C. (2009). Effective short-term opponent exploitation in simplified poker. *Machine Learning*. doi:10.1007/s10994-008-5091-5 (volume and pages unconfirmed).
- ✓ Szafron, D., Gibson, R. & Sturtevant, N. (2013). A parameterized family of equilibrium profiles for three-player Kuhn poker. *AAMAS* (pages unconfirmed).
- ⚠ Tammelin, O. (2014). Solving large imperfect information games using CFR+. arXiv:1407.5042.
- ✓ Zinkevich, M., Johanson, M., Bowling, M. & Piccione, C. (2008). Regret minimization in games with incomplete information. *NIPS 20*, 1729–1736.

---

## Appendix: reproduction

`rnr_package/`: `tree_engine.py` (build and verify the engine), `rnr_general.py` (restricted
responses and linear baselines), `run_queue.py` (all jobs), `summarize.py`, `seed_p0_cache.py`,
`gaps_check.py` (misspecification and both-adapt). Earlier analyses: `alpha_study/`.
Results: 241 restricted-response solves and 64 linear baselines, all on the 5,000-iteration
CFR+ solution.
