import pandas as pd
import matplotlib.pyplot as plt

def prob_plot(preds):
    df = pd.DataFrame({
        'breed': [breed[0] for pred in preds for breed in pred],
        'prob': [breed[1] for pred in preds for breed in pred]
    }).set_index('breed')

    df.sort_values(['prob']).plot.barh(figsize=(8, 3))
    plt.show()
    
def get_key_by_value(d, value):
    for k, v in d.items():
        if v == value:
            return k
    