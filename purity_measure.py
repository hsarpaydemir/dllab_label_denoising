from sklearn.metrics import f1_score

y_true = {"1": [1, 2, 3], "2": [4, 5], "3": [6, 7, 8, 9]}
y_pred = [1, 2, 3, 4, 5, 6, 7, 8, 9]

y_true = sum(y_true.values(), [])
print(y_true)

f1_result = f1_score(y_true, y_pred, average = "macro")
print(f1_result)