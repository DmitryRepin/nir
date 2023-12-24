from fasttext import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, RocCurveDisplay, precision_recall_curve
import pandas as pd
import numpy as np

df = pd.read_csv(r'for_train/trigger_warnings.test', names=['labeled_text'])
model = load_model('model.bin')
nmb_class = 4

def extract_label(text):
    label = text.split()[0]
    return label

def en_to_ru(label):
    if label == '__label__abuse':
        label = 'Абьюз'
    if label == '__label__torture':
        label = 'Пытки'
    if label == '__label__animal_cruelty':
        label = 'Жестокое обращение\n с животными'
    if label == '__label__murder':
        label = 'Убийство'
        
    return label

def probs(pr):
    probs = {pr[0][i] : pr[1][i] for i, _ in enumerate(pr[0])}
    
    return probs

# predict the data
df["predicted"] = df["labeled_text"].apply(lambda x: model.predict(x)[0][0])
df['label'] = df['labeled_text'].apply(extract_label)
# Create the confusion matrix
df['predicted'] = df['predicted'].map(en_to_ru)
df['label'] = df['label'].map(en_to_ru)
labels = df.label.unique()
cm = confusion_matrix(df["label"], df["predicted"], labels=labels)
fig, ax = plt.subplots(figsize=(7,7))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(ax=ax, xticks_rotation='vertical')
plt.legend(loc=0)
plt.savefig('confusion_matrix.png', bbox_inches='tight')
plt.clf()

#pr_r_f1
pr_r_f1 = pd.DataFrame()
precision = []
recall = []
f1 = []
pr_r_f1['Название класса'] = labels
pr_r_f1['Название класса'] = pr_r_f1['Название класса'].map(en_to_ru)
for i in range(cm.shape[0]):
    precision.append(cm[i, i] / np.sum(cm, axis=0)[i])
    recall.append(cm[i, i] / np.sum(cm, axis=1)[i])
    f1.append(2 * precision[i] * recall[i] / (precision[i] + recall[i]))
    
pr_r_f1['Precision'] = precision
pr_r_f1['Recall'] = recall
pr_r_f1['f1-мера'] = f1
print(pr_r_f1.head(4))
pr_r_f1.to_csv('prf.csv')

#roc-curve
df['probs'] = df['labeled_text'].apply(lambda x: model.predict(x, k=nmb_class))
df['probs'] = df['probs'].apply(probs)

df['pr_abuse'] = df['probs'].apply(lambda x: x['__label__abuse'])
df['pr_animal_cruelty'] = df['probs'].apply(lambda x: x['__label__animal_cruelty'])
df['pr_murder'] = df['probs'].apply(lambda x: x['__label__murder'])
df['pr_torture'] = df['probs'].apply(lambda x: x['__label__torture'])

df['abuse'] = df['label'].apply(lambda x: x == 'Абьюз').astype(np.int8)
df['animal_cruelty'] = df['label'].apply(lambda x: x == 'Жестокое обращение\n с животными').astype(np.int8)
df['murder'] = df['label'].apply(lambda x: x == 'Убийство').astype(np.int8)
df['torture'] = df['label'].apply(lambda x: x == 'Пытки').astype(np.int8)

#all roc curves
fpr, tpr, thresholds = roc_curve(df['animal_cruelty'], df['pr_animal_cruelty'])
roc_auc = round(auc(fpr, tpr), 4)
plt.plot(fpr, tpr, label='Абьюз, AUC='+str(roc_auc))

fpr, tpr, thresholds = roc_curve(df['abuse'], df['pr_abuse'])
roc_auc = round(auc(fpr, tpr), 4)
plt.plot(fpr, tpr, label='Жестокое обращение\n с животными, AUC='+str(roc_auc))

fpr, tpr, thresholds = roc_curve(df['torture'], df['pr_torture'])
roc_auc = round(auc(fpr, tpr), 4)
plt.plot(fpr, tpr, label='Пытки, AUC='+str(roc_auc))

fpr, tpr, thresholds = roc_curve(df['murder'], df['pr_murder'])
roc_auc = round(auc(fpr, tpr), 4)
plt.plot(fpr, tpr, label='Убийства, AUC='+str(roc_auc))

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc=0)
plt.savefig('all_roc.png', bbox_inches='tight')
plt.clf()

#abuse roc curve
fpr, tpr, thresholds = roc_curve(df['abuse'], df['pr_abuse'])
roc_auc = round(auc(fpr, tpr), 4)
display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='Абьюз')
display.plot()
plt.legend(loc=0)
plt.savefig('abuse_roc.png', bbox_inches='tight')
plt.clf()

#animal cruelty roc curve
fpr, tpr, thresholds = roc_curve(df['animal_cruelty'], df['pr_animal_cruelty'])
roc_auc = round(auc(fpr, tpr), 4)
display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='Жестокое обращение\n с животными')
display.plot()
plt.legend(loc=0)
plt.savefig('animal_cruelty_roc.png', bbox_inches='tight')
plt.clf()

#torture roc curve
fpr, tpr, thresholds = roc_curve(df['torture'], df['pr_torture'])
roc_auc = round(auc(fpr, tpr), 4)
display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='Пытки')
display.plot()
plt.legend(loc=0)
plt.savefig('torture_roc.png', bbox_inches='tight')
plt.clf()

#murder roc curve
fpr, tpr, thresholds = roc_curve(df['murder'], df['pr_murder'])
roc_auc = round(auc(fpr, tpr), 4)
display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='Убийства')
display.plot()
plt.legend(loc=0)
plt.savefig('murder_roc.png', bbox_inches='tight')
plt.clf()

#precision recall curve abuse
precision, recall, thresholds = precision_recall_curve(df['abuse'], df['pr_abuse'])
plt.plot(precision, recall, label='Абьюз')
plt.legend(loc=0)
plt.xlabel('Precision')
plt.ylabel('Recall')
plt.savefig('abuse_pr_r.png', bbox_inches='tight')
plt.clf()

#precision recall curve animal cruelty
precision, recall, thresholds = precision_recall_curve(df['animal_cruelty'], df['pr_animal_cruelty'])
plt.plot(precision, recall, label='Жестокое обращение\n с животными')
plt.legend(loc=0)
plt.xlabel('Precision')
plt.ylabel('Recall')
plt.savefig('animal_cruelty_pr_r.png', bbox_inches='tight')
plt.clf()

#precision recall curve murder
precision, recall, thresholds = precision_recall_curve(df['murder'], df['pr_murder'])
plt.plot(precision, recall, label='Убийство')
plt.legend(loc=0)
plt.xlabel('Precision')
plt.ylabel('Recall')
plt.savefig('murder_pr_r.png', bbox_inches='tight')
plt.clf()

#precision recall curve torture
precision, recall, thresholds = precision_recall_curve(df['torture'], df['pr_torture'])
plt.plot(precision, recall, label='Пытки')
plt.legend(loc=0)
plt.xlabel('Precision')
plt.ylabel('Recall')
plt.savefig('torture_pr_r.png', bbox_inches='tight')
plt.clf()

#precision recall curve
precision, recall, thresholds = precision_recall_curve(df['abuse'], df['pr_abuse'])
plt.plot(precision, recall, label='Абьюз')

precision, recall, thresholds = precision_recall_curve(df['animal_cruelty'], df['pr_animal_cruelty'])
plt.plot(precision, recall, label='Жестокое обращение\n с животными')

precision, recall, thresholds = precision_recall_curve(df['murder'], df['pr_murder'])
plt.plot(precision, recall, label='Убийство')

precision, recall, thresholds = precision_recall_curve(df['torture'], df['pr_torture'])
plt.plot(precision, recall, label='Пытки')

plt.xlabel('Precision')
plt.ylabel('Recall')
plt.legend(loc=0)
plt.savefig('all_pr_r.png', bbox_inches='tight')
plt.clf()
