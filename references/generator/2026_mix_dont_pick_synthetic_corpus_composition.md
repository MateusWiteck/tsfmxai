# Mix, Don't Pick: Why Synthetic Corpus Composition Matters for Time Series Foundation Model Pretraining

- Authors: Aaryan Nagpal, Debdeep Sanyal, Murari Mandal, Dhruv Kumar, Saurabh Deshpande.
- Year: 2026. arXiv:2606.09912, version 1.
- Source: https://arxiv.org/abs/2606.09912
- Local PDF: `2026_mix_dont_pick_synthetic_corpus_composition.pdf`.
- Citation key: `nagpal2026mixdontpick`.

## Relevance

The authors train forecasting models from random initialization on separately constructed synthetic corpora and evaluate them zero-shot on shared real tasks. This motivates our controlled-corpus comparison. Their setting is univariate and their models differ from our multivariate Chronos-2; our protocol is an adaptation.

## Important evidence

- PDF p. 2, Section 2: the study compares 11 generator families across two model architectures, with zero-shot evaluation on GIFT-Eval.
- PDF p. 12, Appendix C, Table 16: every generator-specific corpus contains one million univariate windows of length 1024. This supports the claim that source-corpus size is controlled across generator conditions.
- PDF p. 14, Appendix E, Table 19: the listed models are initialized without pretrained weights and trained for 50,000 steps with a linear learning-rate schedule. These are the authors' settings, not defaults adopted for our Chronos-2.

## Limitation for our study

Equal corpus size answers a question about data utility under controlled exposure. It does not measure the total cost of generating each corpus.
