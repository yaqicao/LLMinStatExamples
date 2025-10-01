import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False 

model_name = "./biobert-local"
file_path = "note.xlsx"

try:
    df_raw = pd.read_excel(file_path)
    df = df_raw[['note', 'department']].dropna()
    df.columns = ['patient_note', 'department']
    texts = df['patient_note'].tolist() 
    department_codes = sorted(df['department'].unique()) 
    department_to_label = {code: i for i, code in enumerate(department_codes)}
    
    department_names = ['内科', '外科'] 
    label_to_department_name = {i: f'{department_names[i]}' for code, i in department_to_label.items()}
    labels = df['department'].map(department_to_label).values
    

except FileNotFoundError:
    print(f"[错误]文件未找到！检查 '{file_path}' 文件与Python脚本在同一个文件夹下。")
    exit()
except KeyError:
    print(f"[错误]找不到 'note' 或 'department' 列。")
    exit()

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


def get_embeddings(sentences, tokenizer, model):
    sentences_str = [str(s) for s in sentences]
    inputs = tokenizer(sentences_str, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.pooler_output
    return embeddings.numpy()

X = get_embeddings(texts, tokenizer, model)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=30, random_state=42)
X_pca = pca.fit_transform(X_scaled)
X_train, X_test, y_train, y_test = train_test_split(X_pca, labels, test_size=0.25, random_state=42, stratify=labels)

#  C=100
final_classifier = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced', C=100)
final_classifier.fit(X_train, y_train)

y_pred = final_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report_names = [label_to_department_name[i] for i in sorted(label_to_department_name.keys())]
report = classification_report(y_test, y_pred, target_names=report_names)

print(f"准确率 (Accuracy): {accuracy:.4f}")
print("分类报告 (Classification Report):")
print(report)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(12, 8))
colors = ['#1f77b4', '#ff7f0e'] 

for label_num, department_name in label_to_department_name.items():
    plt.scatter(X_pca[labels == label_num, 0], 
                X_pca[labels == label_num, 1], 
                c=colors[label_num], 
                label=department_name,
                alpha=0.8)

plt.title('BioBERT病历文本分类可视化', fontsize=20)
plt.xlabel('主成分 1', fontsize=16)
plt.ylabel('主成分 2', fontsize=16)
plt.legend(title="科室类别", fontsize=14, title_fontsize=16) 
plt.grid(True)
plt.savefig("icu_prediction_plot.svg", format="svg", bbox_inches='tight')
plt.show() 
