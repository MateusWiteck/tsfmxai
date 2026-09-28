Three different questions
1. What rule governs the series?
What mathematical mechanism produces the next value?
This is system identification.
Example:
load(t+1) =
0.7 × load(t)
+ 0.2 × temperature(t)
+ seasonal_effect(t)
2. Which variables influence each other?
temperature(t-2) → load(t)
weekday(t)       → load(t)
load(t-1)        → load(t)
This is causal discovery.
3. Why did a model produce this forecast?
forecast model → SHAP values
This is model explanation.
The downloaded papers mainly address questions 1 and 2. Their learned mechanisms can then become the function used for question 3.