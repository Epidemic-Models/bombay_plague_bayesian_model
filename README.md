# Bayesian Mechanistic Model of the Bombay Plague Epidemic

This repository contains the code for a mechanistic and Bayesian analysis
of the Bombay plague epidemic.

The model represents plague transmission using a delay-SEIR framework and
separates the force of infection into two components:

- climate-dependent reservoir/spillover transmission, `beta_R(t)`
- human-associated transmission, `beta_H(t)`

The total force of infection is

lambda(t) = beta_R(t) + beta_H(t) I(t) / N.

## Human transmission models

Four alternative formulations of `beta_H(t)` are compared:

1. **Constant**
   - human-associated transmission remains constant through time.

2. **Sigmoid**
   - human-associated transmission changes systematically through epidemic time.

3. **Humidity-only**
   - human-associated transmission depends on vapour pressure.

4. **All-climate**
   - human-associated transmission depends on temperature, vapour pressure,
     and rainfall.

## Bayesian inference

Parameters are estimated using Bayesian nested sampling implemented with
`dynesty`.

The mortality observation model uses a negative-binomial likelihood.

Nested sampling is used to obtain:

- posterior parameter distributions
- uncertainty in model predictions
- Bayesian evidence, `log(Z)`
- comparison of alternative `beta_H(t)` formulations

## Project structure

- `data/` — raw plague and climate data
- `prepared_data/` — processed and aligned weekly data
- `src/` — mechanistic model, climate model, priors, likelihood, and observation model
- `scripts/` — data preparation, nested sampling, posterior analysis, and plotting
- `results/` — Bayesian inference outputs and figures

## Main workflow

Prepare the data:

    python scripts/prepare_data.py

Run nested sampling for all transmission models:

    python scripts/run_all_models.py

Analyse the posterior distributions:

    python scripts/posterior_analysis.py

Generate posterior predictive results:

    python scripts/posterior_predictive.py

Generate model-fit figures:

    python scripts/plot_model_fits.py