from gensim.models import Word2Vec
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from adjustText import adjust_text

sentences = [
    ["患者", "因", "急性", "阑尾炎", "行", "腹腔镜", "切除术"],
    ["老年", "男性", "突发", "胸痛", "心电图", "提示", "心肌梗死"],
    ["术后", "病理", "证实", "为", "肺腺癌", "需", "行", "靶向治疗"],
    ["头部", "磁共振", "检查", "发现", "颅内", "存在", "占位性病变"],
    ["患者", "右腿", "骨折", "术后", "康复", "顺利", "功能", "恢复", "良好"]
]

model = Word2Vec(
    sentences=sentences, vector_size=100, window=5, min_count=1, workers=4,
    epochs=5000, sg=1, hs=0, negative=5, seed=42
)

words = list(model.wv.index_to_key)
vectors = np.array([model.wv[word] for word in words])
pca = PCA(n_components=2)
result = pca.fit_transform(vectors)

fig, (ax_left, ax_right) = plt.subplots(1, 2, sharey=True, facecolor='w', 
                                        figsize=(14, 8), 
                                        gridspec_kw={'width_ratios': [1, 1.5], 'wspace': 0.05})

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
ax_left.set_xlim(-1.7, -1.2)
ax_right.set_xlim(0.5, 1.5)
for ax in [ax_left, ax_right]:
    ax.scatter(result[:, 0], result[:, 1], s=100, alpha=0.8, edgecolors='k', c='skyblue')
    ax.grid(True, linestyle='--', alpha=0.4)
    texts = []
    for i, word in enumerate(words):
        x, y = result[i, 0], result[i, 1]
        if ax.get_xlim()[0] <= x <= ax.get_xlim()[1]:
            texts.append(ax.text(x, y, word, fontsize=14))
    
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="->", color='darkred', lw=1))

ax_left.spines['right'].set_visible(False)
ax_right.spines['left'].set_visible(False)
ax_right.yaxis.set_visible(False) 
d = .015 
kwargs = dict(transform=ax_left.transAxes, color='k', clip_on=False)
ax_left.plot((1-d, 1+d), (-d, +d), **kwargs) 
ax_left.plot((1-d, 1+d), (1-d, 1+d), **kwargs) 

kwargs.update(transform=ax_right.transAxes) 
ax_right.plot((-d, +d), (-d, +d), **kwargs) 
ax_right.plot((-d, +d), (1-d, 1+d), **kwargs) 

fig.suptitle("Word2Vec词向量空间", fontsize=22, y=0.96)
fig.text(0.5, 0.04, '主成分一', ha='center', fontsize=18)
ax_left.set_ylabel("主成分二", fontsize=18)

plt.savefig("word_vectors_broken_axis.png", dpi=300, bbox_inches='tight')
plt.show()