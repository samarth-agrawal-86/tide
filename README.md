# Using the model

## Files provided in the package
1. requirements.txt
2. model_training.ipynb
3. lgb_model2.joblib
4. model_predictions.py
5. config.py

# To generate predictions
* Success of the code will be determined by the column `pred_prob_rank`
	* Rank = 0, means no match found
	* Rank = 1, means the highest probability of the receipt to match
	* Subsequently, ranks are provided till 5 Ranks. Higher the rank lower the probability from the model
* Following code will generate predictions for the input data

```python
# load libraries
import pandas as pd
from model_prediction import generate_prediction
from config import THRESHOLD, MAX_RANKS, FEATURES

# load input data
input_df = pd.read_csv('file.csv', sep=':')

# use prediction function to generate output
preds = generate_prediction(data = input_df, threshold=THRESHOLD, max_ranks=MAX_RANKS, features=FEATURES)
```
