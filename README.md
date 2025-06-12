# RCDyM
Reservoir-Computing-Based Dynamical Measure

This repository contains the code for the paper:
- [Ultra-Early Prediction of Tipping Points: Integrating Dynamical Measures with Reservoir Computing]

In this work, we propose a model-free and lightweight machine learning technique, termed “ RC-based dynamical measure” (RCDyM) method, which can effectively estimate latent and stability-related dynamical measures. By analyzing temporal trends in these dynamical measures, our approach enables ultra-early prediction of tipping points. 

This respository provides RCDyM methods implemented in PyTorch. And the experimental data can be generated through code or downloaded from the corresponding dataset website.

## Environment 
To run this project, we will need to set up an environment with Python 3 and install the following Python packages:
- Python 3.11.8
- joblib 1.4.2
- matplotlib 3.9.0
- numpy 1.26.4
- pandas 2.2.2
- scipy 1.14.1
- scikit-learn 1.5.2
- torch 2.2.2
- torchdiffeq 0.2.4

```python
pip install -r requirements.txt
```

## Examples
- 'bifurcation_fold.py' serves as an example for predicting tipping point in a fold bifurcation system, corresponding to the transition between equilibrium states of the system.
```python
python -m bifurcation_fold
```
- 'floquent_logistic.py' serves as an example for predicting tipping points of periodic variations in the logistic map, corresponding to the transformation of the system's periodicity.
```python
python -m floquent_logistic
```

## Files
- 'bifurcation_fold.py' is experimental code for generating fold bifurcation data and performing critical transition prediction.
- 'bifurcation_hopf.py' is experimental code for generating Hopf bifurcation data and performing critical transition prediction.
- 'bifurcation_PeriodD.py' is experimental code for generating period-doubling bifurcation data and performing critical transition prediction.
- 'bifurcation_pitchfork.py' is experimental code for generating pitchfork bifurcation data and performing critical transition prediction.
- 'floquent_logistic.py' is experimental code for generating the logistic map data and performing critical transition prediction.
- 'hopf_pf.py' is experimental code for generating the hopf data and performing critical transition prediction (from period to fixed point).

## File folders
- 'utils' folder contains the code for RCDyM method.
- 'models' folder: used to store model files
- 'dataset' folder: used to store dataset files
- 'real_data' folder: used to store real data files
- 'results' folder: used to store results files
- 'RD_experiments' folder: used to execute experimental code for real-world data
- 'Robustness_analysis' folder: used to execute experimental code for robustness analysis
