from sklearn.metrics import f1_score

#noisy, clean, and output of plc needed
y_true = {"1": [1, 2, 11], "2": [4, 5], "3": [6, 7, 8, 9]}
y_pred = [1, 2, 3, 4, 5, 6, 7, 8, 9]

y_true = sum(y_true.values(), [])
print(y_true)

f1_result = f1_score(y_true, y_pred, average = "macro")
print(f1_result)

#Intersection over Union Method

#change implementation, order of labels is important
def iou(noisy_labels, updated_labels):
    intersection_labels = set(noisy_labels).intersection(updated_labels)
    union_labels = list(set(noisy_labels) | set(updated_labels))
    print(intersection_labels)
    print(union_labels)
    return len(intersection_labels) / len(union_labels)
iou_labels = iou(y_true, y_pred)
print(iou_labels)

#f1_score my implementation (tp / (tp + 1/ 2 * (fp + fn)))
#compute tp, fp, etc, then compute precision and recall seperately. Be careful of corner cases. 