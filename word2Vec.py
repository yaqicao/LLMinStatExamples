from gensim.models import Word2Vec
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from adjustText import adjust_text
import matplotlib.font_manager as fm

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

try:
    similarity = model.wv.similarity('胸痛', '心肌梗死')
    print(f"\n\"胸痛\"与\"心肌梗死\"的余弦相似度: {similarity:.4f}")
    similar_words = model.wv.most_similar('病理', topn=2)
    print("\n与\"病理\"最相似的2个词:")
    for word, score in similar_words:
        print(f"  - {word}: {score:.4f}")
    odd_word = model.wv.doesnt_match(["腹腔镜", "切除术", "靶向治疗", "心电图"])
    print(f"\n在 [腹腔镜, 切除术, 靶向治疗, 心电图] 中最不匹配的词是: {odd_word}")
   
    word_to_check = '患者'
    word_vector = model.wv[word_to_check]
    first_five_dims = word_vector[:5]
    np.set_printoptions(formatter={'float': '{: 0.3e}'.format})
    print(f"\n词语 \"{word_to_check}\" 的词向量:")
    print(f"其前五维的数值分别是：{first_five_dims}")
    np.set_printoptions()


except KeyError as e:
    print(f"\n词{e}不在词汇表中")

try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'FangSong', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    words = list(model.wv.index_to_key)
    vectors = np.array([model.wv[word] for word in words])
    pca = PCA(n_components=2)
    result = pca.fit_transform(vectors)
    plt.figure(figsize=(14, 14))
    plt.scatter(result[:, 0], result[:, 1], s=100, alpha=0.8, edgecolors='k')
    texts = []
    for i, word in enumerate(words):
        texts.append(plt.text(result[i, 0], result[i, 1], word, fontsize=18))

    adjust_text(texts,
                lim=4000,
                force_points=(5.0, 6.0),
                force_text=(4.0, 5.0),
                expand_points=(1.8, 1.8),
                arrowprops=dict(arrowstyle="->", color='darkred', lw=1.2,
                                shrinkA=8, shrinkB=8)
               )


    plt.title("Word2Vec词向量空间 ", fontsize=28)
    plt.xlabel("主成分一", fontsize=22)
    plt.ylabel("主成分二", fontsize=22)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig("word_vectors.svg", format='svg', bbox_inches='tight')

   
except ImportError:
    print("\n缺失相关库")
#except Exception as e:
    #print(f"\n可视化过程中出现错误: {e}")