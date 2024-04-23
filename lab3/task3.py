import numpy as np
import matplotlib.pyplot as plt

from skimage import io
from skimage.color import rgb2lab, lab2rgb
from sklearn.cluster import KMeans

img = io.imread("lana.jpg")

plt.imshow(img)
plt.gca().axis("off")
plt.show()


lab_img = rgb2lab(img)
flat_img = lab_img.reshape((-1, 3))
print(flat_img.shape)
i = 2
while i <= 128:
    km = KMeans(n_clusters=i)
    km.fit(flat_img)
    labels = km.predict(flat_img)

    km_img = np.apply_along_axis(lambda x: km.cluster_centers_[x], 0, labels)

    # преобразуем изображение обратно к трехмерному массиву
    km_img = km_img.reshape(img.shape)

    pics = plt.figure(figsize=(8, 8))
    plt.imshow(lab2rgb(km_img))
    plt.gca().axis("off")
    plt.title("Кол-во кластеров: {0}".format(i))
    plt.savefig("lana_{0}.png".format(i))
    plt.close()
    i*=2