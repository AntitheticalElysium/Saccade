# Printed outputs of the v1 diagnosis (2026-11-24 ~02:00-02:30), run against memory_v1.py (= memory.py @ 54d2c13b) and
# run/localizer_v1.pt, on the first N panel episodes (tokens_panel.pt). Scripts in this folder.

## localizer_diagnosis.py 2  -> localizer_diagnosis_2eps.json (moved n=461: prior 0.716, posterior 0.701, match alone 0.714;
##   wrong match = stay 0.788, wrong has more overlap 0.614; blocked n=80: posterior 0.988)

## percell.py (3 episodes; moved steps; mean (stay - true) per map cell, 7 x 9; centre = row 3 col 4)
[[ -5.8 -13.1 -13.1 -12.2 -11.5 -12.8 -14.5 -14.3  -8.5]
 [ -9.6 -15.9 -17.2 -17.  -16.4 -15.3 -17.7 -17.6 -10.8]
 [ -9.7 -15.4 -13.6 -14.   -1.6 -15.7 -17.5 -16.8  -9.8]
 [ -9.6 -15.5 -14.1   1.9  32.7   1.6 -14.7 -14.   -8.2]
 [ -9.1 -15.1 -15.6 -12.2   1.8 -12.5 -13.1 -13.6  -7.7]
 [ -9.2 -14.7 -15.8 -13.6 -13.3 -13.7 -13.  -12.9  -8.2]
 [ -4.6  -8.9 -10.3  -9.9  -8.5  -8.   -8.6  -7.5  -2.1]]
P(stay beats true) centre 0.5, elsewhere 0.1-0.3; total mean advantage of stay -598.3

## apportion.py (4 episodes): moved steps 947, match wrong 249
right_without_center_stays_right 0.996
wrong_choice_stay 0.807
true_wins_without_center 0.068
true_wins_without_center_plus4 0.197
true_wins_with_prior 0.068
true_wins_with_prior_without_center 0.145
true_wins_common_cells_without_center 0.189
gap_choice_minus_true 130.619        (mean nats, wrong steps)
center_contrib 30.459
noncommon_contrib 63.113
learned scale b by age 1..256: 0.234 .. 0.446; trust 0.655 .. 0.612; b_bg 0.764

## counterfactual.py (6 episodes): match alone, current / coverage-neutral / sprite removed / both
moved_day   782: 0.890 / 0.935 / 0.903 / 0.940
moved_dark  526: 0.399 / 0.407 / 0.422 / 0.422
blocked_day 154: 0.708 / 0.721 / 0.708 / 0.727
blocked_dark 74: 0.797 / 0.797 / 0.784 / 0.797

## counterfactual2.py (same 6 episodes): with the prior, central cells, prior alone
moved_day:  both+prior 0.957, prior alone 0.932, central(cheb<=2) 0.881
moved_dark: both+prior 0.411, prior alone 0.384, central(cheb<=2) 0.411, central(cheb<=1) 0.291
blocked_day: both+prior 0.981, prior alone 0.909; blocked_dark: both+prior 0.959, prior alone 0.919
(the 'command_only' column is a mapping typo in the script and is void)

## Classic's renderer (craftax_classic/renderer.py:408-439, constants.py:385-399): below daylight 0.5, per-pixel static
## of intensity 2(0.5 - daylight) * (1 - exp(-r^2 / (2 * 0.5^2))), r = distance from the screen centre in [-1, 1] units
