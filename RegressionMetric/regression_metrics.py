from sklearn.metrics import r2_score

def adjusted_r2(data,y_true,y_pred):
    r2 = r2_score(y_true,y_pred)
    n = data.shape[0]
    p = data.shape[1]

    adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)

    return adjusted_r2