# Audit of the 26th Knesset election (Israel, 27 October 2026) with public data

Forensic tests on the results by ballot box (קלפי) published by the Central Elections Committee, compared with the 2022 election (25th Knesset). Article: [lilianezukerman.com/research/israel-2026-election-audit-public-data.html](https://lilianezukerman.com/research/israel-2026-election-audit-public-data.html).

## Data

Download the file "Results by Voting Station" from each results site and save it in `dados/`:

| File | Source |
|---|---|
| `dados/expb.csv` | votes26.bechirot.gov.il (2026) |
| `dados/expb_2022.csv` | votes25.bechirot.gov.il (2022) |
| `dados/expc.csv` | votes26.bechirot.gov.il (2026), "Results by locality", used only by the map |

`dados/expb.csv` is the final file, published by the committee on 4 November 2026. Between the file of 2 November and the final one, 78 boxes changed and 3 double-envelope batches were added (list in `dados/diff_expb_2026-11-02_2026-11-04.txt`): the 11 boxes flagged as transcription errors (9 by the 15% rule, 2 by the 3% rule) were all corrected in the predicted direction, and the committee corrected another 67 slips between neighbouring letters of 1 to 27 votes that the thresholds here did not reach. In all, 1,789 votes moved or added out of 5.7 million; no seat changes and no test result moves beyond the second decimal place. Earlier versions: between the file of 30 October and that of 2 November, 8 boxes changed (list in `dados/diff_expb_2026-10-30_2026-11-02.txt`): 1 of the 9 boxes flagged as transcription errors was corrected (Idan 1), plus 3 slips of the same kind below the thresholds here and 4 letter mix-ups of up to 13 votes. National effect: 155 votes from רץ to רק and one more voter; no seat changes. Boxes with `סמל ישוב` = 9999 are double envelopes (soldiers, diplomats, hospitals), with no electorate of their own; they are left out of the turnout tests.

`dados/levant_outlines.json` holds the country outlines (Natural Earth 10m), used only by the map. `dados/bycode2021.xlsx` is the locality list of the Central Bureau of Statistics (CBS), copied from the R package [il.cbs.muni](https://github.com/matanhakim/il.cbs.muni). It gives the religion (Jewish, Arab, Bedouin tribe, mixed city) and the type of each locality, by the same code the elections committee uses. Druze, Christian and non-tribal Bedouin localities are identified by lists inside the script.

`parties.csv` gives the English name of each 2026 party by its ballot letters.

## How to run

```bash
pip install pandas numpy scipy openpyxl cvxpy

python3 forensics.py dados/expb.csv dados/boxes.pkl            # 2026: impossible values, digits, turnout, Klimek, invalid ballots
python3 sim_null.py dados/boxes.pkl                                   # 2026: digits and turnout against a binomial simulation
python3 forensics.py dados/expb_2022.csv dados/boxes_2022.pkl  # same tests on 2022, as the reference
python3 sim_null.py dados/boxes_2022.pkl
python3 compare_2022.py                                               # box by box and city by city, by bloc
python3 arab_boxes.py                                               # Arab-majority boxes, 2022 vs 2026
python3 haredi_boxes.py                                               # Haredi-majority boxes, 2022 vs 2026
python3 envelopes.py                                                  # double-envelope boxes (soldiers etc.), 2022 vs 2026
python3 religious_groups.py                                           # tests by religious group / community, 2022 vs 2026
python3 transfer.py locality                                          # vote transfer 2022 -> 2026 (by locality; 'box' for paired boxes)
python3 more_findings.py                                              # seats, scenario without the Arab surge, where the new voters came from, border communities
python3 extra_tests.py                                                # invalid 2022 vs 2026, invalid vs party, integer percentages, turnout change by box
python3 invalid_profile.py                                            # invalid ballots vs city profile (CBS: age, schooling, socio-economic level)
python3 same_site.py dados/expb.csv                                   # each box vs the others at the same site; lists likely protocol transcription errors
python3 fringe_check.py                                               # second transcription-error detector (micro-lists with many votes in one box)
python3 slip_check.py dados/expb.csv                                  # specific detector for the neighbouring-row slip on the protocol (רץ/רק etc.), lower thresholds
python3 committee_check.py                                            # by regional committee (data-entry centre): digits and invalid ballots, 2026 vs 2022
python3 map_localities.py                                             # map: first place and bloc majority by locality (needs dados/expc.csv)
python3 diff_results.py dados/expb_preliminary.csv dados/expb.csv        # what changed between two downloads (preliminary vs final)
```

## Notes on method

- The column `ריכוז` groups the boxes of one polling site. `same_site.py` compares each box with the others at the same site and looks for the transcription-error pattern: a party strong at the site disappears from one box, and the same number of votes appears under a list with almost no votes there. The list is written to `dados/likely_transcription_errors.csv`.
- Double-envelope boxes are filled in batches (about 600 envelopes in 2022 and 500 in 2026), so the digits of their voter totals are not tested.
- Digit tests: the main reference is `sim_null.py`, which redraws each count as Binomial(valid votes, observed share) and compares the observed chi-square with the simulated one. The smoothed baseline of `forensics.py` rejects often for small parties in both years, a limitation of that method.
- The parties changed between 2022 and 2026; the comparison uses blocs: arab, haredi, netanyahu_coalition and likud.
- Only part of the boxes keep the same number between elections; the comparison by city covers the rest.

## Corrections and suggestions

eleicoes@lilianezukerman.com

This analysis was done with the help of an AI model (Claude, by Anthropic); every result was checked against the published files, and the scripts are here so that nobody has to trust either of us.
