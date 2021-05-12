from sklearn.metrics import auc, roc_auc_score
all_labels =  list(class_names.keys())

def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)

    for (idx, c_label) in tqdm(enumerate(all_labels)): # all_labels: no of the labels, for ex. ['cat', 'dog', 'rat']
        fpr, tpr, thresholds = roc_curve(y_test[:,idx].astype(int), y_pred[:,idx])
        c_ax.plot(fpr, tpr, label = '%s (AUC:%0.2f)'  % (c_label, auc(fpr, tpr)))
    c_ax.plot(fpr, fpr, 'b-', label = 'Random Guessing')
    plt.legend()
    plt.grid(True)
    return roc_auc_score(y_test, y_pred, average=average)
  
  fig, c_ax = plt.subplots(1,1, figsize = (12, 8))


validation_generator.reset() # resetting generator
y_pred = classifier.predict_generator(validation_generator, verbose = True)
y_pred = np.argmax(y_pred, axis=1)
multiclass_roc_auc_score(validation_generator.classes, y_pred)
