import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.manifold import TSNE
from sklearn.neighbors import KernelDensity


exp_name = 'exp_demo_sac_embed_dataset_240112_132144'
episode = 50

data = []
for eps in range(episode):
    data.append(np.load(f'./{exp_name}/embed_obs_box_{eps+1}.npy'))
print(f'obs shape: {data[0].shape}')


# t-sne
checkpoint = []
X = None
for idx, dd in enumerate(data):
    if idx == 0:
        X = dd
    else:
        X = np.concatenate((X, dd), axis=0)
    checkpoint.append(X.shape[0])
print(f'X shape: {X.shape}')

perplexity = [5,10,20,30,50]
for per in perplexity:
    X_embedded = TSNE(n_components=2, learning_rate='auto',
                    init='random', perplexity=per).fit_transform(X)
    print(X_embedded.shape)

    plt.figure()
    last = 0
    for idx, check in enumerate(checkpoint[:10]):
        x, y = X_embedded[last:check, 0], X_embedded[last:check, 1]
        last = check
        plt.scatter(x, y, label=f'episode_{idx}', alpha=0.3, s=10)
    plt.title(f'tsne perplexity_{per}') #设置图名为Simple Plot
    plt.legend() #自动检测要在图例中显示的元素，并且显示
    plt.savefig(f'./{exp_name}/tsne_{per}.png')
    plt.close()


    # KDE
    X = X_embedded[:checkpoint[:10]]
    kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(X)
    kde.score_samples(X)