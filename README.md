# RLEL

Code for WWW 2019 paper: "Joint Entity Linking with Deep Reinforcement Learning"


## Dependencies
This project is based on ```python>=3.6```. The dependent package for this project is listed as below:
```
tensorflow>=1.8.0
scikit-learn==0.21.3
xgboost==0.9
```

## Training Command
1.Extract statistical features
```
python model/local_feature.py
```

2.Calculate xgboost score and filter candidate 
```
python model/xgboost_rank.py
```

3.Train local LSTM model
```
python model/local_model.py
```

4.Rank mention
```
python model/local_ranker.py
```

5.Train global LSTM model
```
python model/global_model.py
```

6.Train policy model
```
python model/policy_model.py
```