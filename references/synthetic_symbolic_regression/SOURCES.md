# Referências sobre descoberta simbólica de mecanismos dinâmicos

Esta pasta reúne trabalhos sobre regressão simbólica, identificação de sistemas e síntese de programas aplicadas a sequências temporais e sistemas dinâmicos. Os PDFs foram verificados quanto à assinatura PDF, abertura, número de páginas e correspondência entre a primeira página e o título esperado.

## Índice dos artigos

| # | Referência | Arquivo local | Fonte aberta | Papel na revisão |
|---:|---|---|---|---|
| 1 | D'Ascoli et al. (2022), *Deep Symbolic Regression for Recurrent Sequences* | [PDF](01_2022_dascoli_deep_symbolic_regression_for_recurrent_sequences.pdf) | [arXiv:2201.04600](https://arxiv.org/abs/2201.04600) | Inferência de uma relação de recorrência simbólica a partir dos termos observados de uma sequência em tempo discreto. |
| 2 | D'Ascoli et al. (2024), *ODEFormer: Symbolic Regression of Dynamical Systems with Transformers* | [PDF](02_2024_odeformer_symbolic_regression_of_dynamical_systems.pdf) | [arXiv:2310.05573](https://arxiv.org/abs/2310.05573) | Pretraining sintético e decodificação direta de equações diferenciais simbólicas a partir de trajetórias. |
| 3 | Şahin, Kilbertus e Becker (2025), *Predicting Symbolic ODEs from Multiple Trajectories* (MIO) | [PDF](03_2025_mio_predicting_symbolic_odes_from_multiple_trajectories.pdf) | [arXiv:2510.23295](https://arxiv.org/abs/2510.23295) | Uso de múltiplas trajetórias produzidas pelo mesmo mecanismo para identificar uma única equação simbólica. |
| 4 | Seifner et al. (2025), *In-Context Learning of Stochastic Differential Equations with Foundation Inference Models* (FIM-SDE) | [PDF](04_2025_fim_sde_in_context_learning_of_stochastic_differential_equations.pdf) | [arXiv:2502.19049](https://arxiv.org/abs/2502.19049) | Inferência amortizada de drift e diffusion a partir de múltiplas realizações estocásticas. |
| 5 | Khandelwal, Schoukens e Tóth (2020), *A Tree Adjoining Grammar Representation for Models of Stochastic Dynamical Systems* | [PDF](05_2020_tree_adjoining_grammar_stochastic_dynamical_systems.pdf) | [arXiv:2001.05320](https://arxiv.org/abs/2001.05320) | Gramática formal para representar modelos dinâmicos estocásticos, incluindo estruturas da família NARMAX. |
| 6 | Khandelwal, Schoukens e Tóth (2023), *Automated Multi-objective System Identification Using Grammar-Based Genetic Programming* | [PDF](06_2023_automated_multi_objective_system_identification_grammar_gp.pdf) | [TU Eindhoven / DOI](https://doi.org/10.1016/j.automatica.2023.111017) | Identificação de sistemas com busca gramatical e otimização simultânea de ajuste aos dados e complexidade estrutural. |
| 7 | Breda et al. (2025), *Sparse Identification of Nonlinear Dynamics for Stochastic Delay Differential Equations* | [PDF](07_2025_sparse_identification_stochastic_delay_differential_equations.pdf) | [arXiv:2508.03040](https://arxiv.org/abs/2508.03040) | Identificação esparsa de drift e diffusion em equações diferenciais estocásticas com atraso. |
| 8 | Derner et al. (2020), *Constructing Parsimonious Analytic Models for Dynamic Systems via Symbolic Regression* | [PDF](08_2020_constructing_parsimonious_analytic_models_symbolic_regression.pdf) | [arXiv:1903.11483](https://arxiv.org/abs/1903.11483) | Identificação simbólica de modelos dinâmicos analíticos parcimoniosos, incluindo formulações em tempo discreto. |
| 9 | Iyer, Boddupalli e Moehlis (2024), *Expressive Symbolic Regression for Interpretable Models of Discrete-Time Dynamical Systems* | [PDF](09_2024_expressive_symbolic_regression_discrete_time_dynamical_systems.pdf) | [arXiv:2406.06585](https://arxiv.org/abs/2406.06585) | Regressão simbólica voltada explicitamente à identificação de mapas e sistemas dinâmicos em tempo discreto. |
| 10 | Schaechtle et al. (2016), *Time Series Structure Discovery via Probabilistic Program Synthesis* | [PDF](10_2016_time_series_structure_discovery_probabilistic_program_synthesis.pdf) | [arXiv:1611.07051](https://arxiv.org/abs/1611.07051) | Representação e descoberta de modelos de séries temporais como programas probabilísticos estruturados. |
| 11 | Saad et al. (2019), *Bayesian Synthesis of Probabilistic Programs for Automatic Data Modeling* | [PDF](11_2019_bayesian_synthesis_probabilistic_programs_automatic_data_modeling.pdf) | [arXiv:1907.06249](https://arxiv.org/abs/1907.06249) | Síntese bayesiana de programas probabilísticos geradores condicionada aos dados observados. |
| 12 | Brunton, Proctor e Kutz (2016), *Discovering Governing Equations from Data by Sparse Identification of Nonlinear Dynamical Systems* (SINDy) | [PDF](12_2016_sindy_discovering_governing_equations_from_data.pdf) | [PNAS / DOI](https://doi.org/10.1073/pnas.1517384113) | Referência fundamental para selecionar, por esparsidade, os termos ativos de uma biblioteca de funções candidatas. |
| 13 | Both et al. (2021), *DeepMoD: Deep Learning for Model Discovery in Noisy Data* | [PDF](13_2021_deepmod_deep_learning_for_model_discovery_in_noisy_data.pdf) | [arXiv:1904.09406](https://arxiv.org/abs/1904.09406) | Combinação de aproximação neural da solução com descoberta esparsa de equações na presença de ruído. |
| 14 | Liu, Zhang e Schaeffer (2024), *PROSE: Predicting Operators and Symbolic Expressions Using Multimodal Transformers* | [PDF](14_2024_prose_predicting_operators_and_symbolic_expressions.pdf) | [arXiv:2309.16816](https://arxiv.org/abs/2309.16816) | Predição conjunta da evolução numérica e da expressão simbólica do operador dinâmico. |
| 15 | Yu, Chatzi e Kissas (2025), *Grammar-Based Ordinary Differential Equation Discovery* (GODE) | [PDF](15_2025_gode_grammar_based_ordinary_differential_equation_discovery.pdf) | [arXiv:2504.02630](https://arxiv.org/abs/2504.02630) | Descoberta de ODEs com espaço de expressões restringido por regras gramaticais. |
| 16 | Yu, Chatzi e Kissas (2026), *Neuro-Symbolic ODE Discovery with Latent Grammar Flow* (LGF) | [PDF](16_2026_lgf_neuro_symbolic_ode_discovery_latent_grammar_flow.pdf) | [arXiv:2604.16232](https://arxiv.org/abs/2604.16232) | Geração neuro-simbólica condicionada por uma representação latente de estruturas gramaticais discretas. |
| 17 | Hübers et al. (2026), *Foundation Inference Models for Ordinary Differential Equations* (FIM-ODE) | [PDF](17_2026_fim_ode_foundation_inference_models_for_ordinary_differential_equations.pdf) | [arXiv:2602.08733](https://arxiv.org/abs/2602.08733) | Inferência amortizada do campo vetorial de ODEs; serve como contraste entre representação funcional neural e equação simbólica. |
| 18 | Faraji e Belardinelli (2026), *Verifier-Guided Model Discovery for Physical Dynamical Systems with Pretrained Symbolic Transformers* | [PDF](18_2026_verifier_guided_model_discovery_pretrained_symbolic_transformers.pdf) | [arXiv:2608.02662](https://arxiv.org/abs/2608.02662) | Geração de candidatos simbólicos seguida de verificação orientada pelo comportamento obtido por simulação. |
| 19 | Breda, Tanveer e Wu (2026), *Sparse Identification of Delay Equations with Distributed Memory* | [PDF](19_2026_sparse_identification_delay_equations_distributed_memory.pdf) | [arXiv:2512.21070](https://arxiv.org/abs/2512.21070) | Identificação de equações com atraso distribuído, representando memória por kernels em vez de somente atrasos pontuais. |

## Notas bibliográficas

- O artigo 2 foi disponibilizado como preprint em 2023 e apresentado na ICLR 2024; o ano 2024 foi mantido no nome do arquivo.
- O artigo 4 foi inicialmente disponibilizado em 2025; o PDF baixado contém uma revisão de 2026.
- O artigo 6 foi publicado na *Automatica*, volume 154, em 2023, artigo 111017. O PDF provém do repositório institucional da Eindhoven University of Technology.
- O artigo 7 aparece no arXiv em 2025. Por isso, o arquivo usa 2025, embora a lista de trabalho o associasse a 2026.
- O artigo 9 tem material preliminar datado de 2023, mas seu registro no arXiv é de 2024.
- O artigo 10 foi submetido ao arXiv em 2016 e revisado em 2017; o ano original foi preservado.
- O artigo 13 foi disponibilizado inicialmente como preprint em 2019 e publicado em periódico em 2021; o nome do arquivo segue a publicação.
- O artigo 14 foi disponibilizado como preprint em 2023 e publicado em 2024; o nome do arquivo segue a publicação.
- O artigo 19 foi submetido ao arXiv no final de 2025 e está associado à publicação de 2026; o nome do arquivo usa 2026.

## Verificação local

- Quantidade: 19 PDFs.
- Integridade: todos começam com a assinatura `%PDF-` e foram processados por `pdfinfo` e `pdftotext`.
- Identidade: os títulos e autores foram conferidos na primeira página renderizada de cada arquivo.
- Paginação observada: entre 5 e 35 páginas por documento.
