import pickle
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

with open('saved_dictionary2.pickle', 'rb') as f:
    loaded_dict = pickle.load(f)

src_f, src_y, trg_f, trg_y, accl = loaded_dict['1'],loaded_dict['2'],loaded_dict['3'],loaded_dict['4'],loaded_dict['acc']

plt.plot(accl)
all_f = np.vstack((src_f, trg_f))

tsne = TSNE(2, verbose=1)
tsne_proj = tsne.fit_transform(all_f)

# Plot those points as a scatter plot and label them based on the pred labels
cmap = cm.get_cmap('tab20')
fig, ax = plt.subplots(figsize=(8,8))
num_categories = 6

for activity in range(num_categories):
    indices = trg_y==activity
    indices2 = src_y==activity
    tsne_src, tsne_trg = tsne_proj[0:len(src_f)],tsne_proj[len(src_f):]
    ax.scatter(tsne_src[indices2,0],tsne_src[indices2,1], s=50,marker='o',c=np.array(cmap(activity)).reshape(1,4), label = activity)
    ax.scatter(tsne_trg[indices,0],tsne_trg[indices,1],s=50,  marker='s',c=np.array(cmap(activity)).reshape(1,4), label = activity)
    
ax.legend(fontsize='large', markerscale=2)
plt.show()