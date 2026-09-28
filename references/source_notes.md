https://github.com/mdhabibi/LIME-for-Time-Series


just to cite papers :

LoMEF: A framework to produce local explanations for global model time series forecasts
    They propose a model agnostic solution
    We train simpler univariate surrogate models that are considered interpretable (e.g., ETS) on the predictions of the GFM on samples within a neighbourhood that we obtain through bootstrapping, or straightforwardly as the one-step-ahead global black-box model forecasts of the time series which needs to be explained. After, we evaluate the explanations for the forecasts of the global models in both qualitative and quantitative aspects such as accuracy, fidelity, stability, and comprehensibility, and are able to show the benefits of our approach.


Faithful and Interpretable Explanations for Complex Ensemble Time Series Forecasts using Surrogate Models and Forecastability Analysis
    The name says that is more focus on the model, but could be good for evaluation


XForecast: Evaluating Natural Language Explanations for Time Series Forecasting
    If the model evolve to a NLP integration 

Interpretable Multivariate Time Series Forecasting with Temporal Attention Convolutional Neural Networks
    model orinted paper
    good for evaluation metrics


Spatiotemporal Attention for Multivariate Time Series Prediction and Interpretation
    model orinted paper
    good for evaluation metrics

ERASER: A Benchmark to Evaluate Rationalized NLP Models
    define sufficiency/comprehensiveness metrics
    