# How real eyes choose where to look, and simpler general forms of Option 3 (2026-11-20)

The user asked: is the world-eye too drawn to unpredictable content, and how does the real eye behave? Is Option 3
hard-coded, and are there simpler, more general ways? This pass draws on abstracts and publisher pages; full texts
were not read.

## The real eye

1. **Surprise is belief change, not unexpectedness.**
   - [Itti & Baldi 2009](https://ilab.usc.edu/publications/Itti_Baldi09vr.html) define surprise as the KL divergence
     between the observer's prior and posterior. "Only data observations which substantially affect the observer's
     beliefs yield surprise, irrespectively of how rare or informative in Shannon's sense".
   - On television clips, 72% of human gaze shifts go to locations more surprising than average (84% where observers
     agree).
   - Noise that changes no belief is unexpected but not surprising: the noisy-TV answer, from vision science.
2. **In natural tasks gaze serves reward and uncertainty, not salience.**
   - [Tatler, Hayhoe, Land & Ballard 2011](https://pmc.ncbi.nlm.nih.gov/articles/PMC3134223): salience "can fail
     dramatically" in natural tasks. Gaze is allocated by "reward maximization and uncertainty reduction".
   - The Sprague & Ballard model, tested by [Sullivan et al. 2012](https://pmc.ncbi.nlm.nih.gov/articles/PMC3587015):
     drivers look at a noisy speedometer more, but only when that task has priority. Uncertainty counts in proportion
     to how much it matters.
3. **Saccades seek information about hypotheses.**
   - [Yang, Lengyel & Wolpert 2016](https://elifesciences.org/articles/12215): scan paths follow Bayesian active
     sensing (about 70% efficient) and gather information 1.8-1.9x better than heuristics.
   - [Najemnik & Geisler 2005](https://www.nature.com/articles/nature03390.pdf): human search is near the ideal
     foveated searcher.
4. **Learners avoid both the predictable and the unlearnable.**
   - [Kidd, Piantadosi & Aslin 2012](https://pmc.ncbi.nlm.nih.gov/articles/PMC3359326): infants look away most from
     the most predictable and the least predictable sequences (the Goldilocks effect).
   - [Poli et al. 2020](https://www.ndcn.ox.ac.uk/publications/1133731): infants' looking follows surprise,
     predictability and learning progress.
   - [Gottlieb & Oudeyer 2018](https://dx.doi.org/10.1038/s41583-018-0078-0): active sampling depends on attitudes
     toward uncertainty, learning progress and types of information.
5. **Gaze anticipates events** (Land & McLeod 2000, read 11-14) and re-enacts in imagination (RESEARCH_OPTION3.md).

**What this means for our eye:**
- Both of our eyes score Shannon-style unexpectedness: the chooser's squared token error and the world-eye's raw
  error. That is exactly what noise maximises.
- The real eye discounts what does not change its beliefs, weights uncertainty by relevance, and prefers what it can
  learn.
- Implementable, general candidates:
  - (a) **Precision-weighted error** (predictive coding; AMA, Mavor-Parker 2022): the error divided by the world's own
    predicted spread, estimable from its sample spread. This is the eye's score only, not the loss weight (Seitzer's
    pitfall, our 10-31 and 11-01 results).
  - (b) **Belief change:** how much the window's content moves the world's state or predictions.
  - (c) **Learning progress** (the user's 11-01 idea).
  - (d) **Value of information** (P3).

## Simpler, general forms of Option 3

- **My 11-20 version is partly hand-coded.** K, refresh-or-replace-the-stalest and the scroll remap are rules.
  `RECORDS.md` v3 is heavier still: learned association over all evidence, lifecycle, revisability, store.
- **What the brain points to:** attention gates what gets encoded.
  - [Aly & Turk-Browne 2016](https://pmc.ncbi.nlm.nih.gov/articles/4743819/): attention promotes episodic encoding by
    stabilising hippocampal representations.
  - The write is privileged; the rest can be generic.
- **General learned mechanisms:**
  - **Selective state spaces** ([Mamba](https://arxiv.org/html/2312.00752v2)): an input-dependent step size Δ
    decides per input whether to overwrite (large Δ) or hold and ignore (small Δ). This is the general form of "write
    when attended, hold otherwise", learned, with no slot count or replacement rule.
  - **[Token Turing Machines](https://arxiv.org/abs/2211.09119v2):** a small external memory of tokens, read and
    written by learned token summarisation each step. Bounded cost; beats long-sequence transformers and RNNs on
    online video detection and robot policies.
  - **[XMem](https://arxiv.org/abs/2207.07115v2):** separate sensory, working and long-term stores with consolidation.
    It wins on long videos and is heavier.
- **Candidates on our own machinery, simplest first:**
  - G1. **Foveal record channel.** The existing parameter-free S4D record, fed only at fixated cells with the glimpse's
    content: "write where you look", no new rule. Read like the record (`record_read`, `record_in`). Limits: a fixed
    decay, and screen-registered (as the record already is).
  - G2. **Selective spatial record.** The same channel with a learned, input-dependent write / hold (a Mamba-style Δ per
    cell; inputs = token, glimpse, fixation flag). The hold becomes learned. Needs a gradient into the record's own
    parameters and a request for it (our 11-16 lesson: learned memory needs an incentive).
  - G3. **Token memory** (TTM-style) written with the glimpse as a privileged input. The most general and the heaviest;
    it overlaps Mamba's memory tokens.
- **All three depend on the eye looking at what matters,** which loops back to the eye's score above.
