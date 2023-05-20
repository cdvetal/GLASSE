
from PIL import Image
main = "./outputs"
tsne_mine = "tsne_mnist_format.png"
tsne_mnist = "tsne_MNIST_dataset.png" 

for i in range(1,16):
	output_dir = f"./outputs/output{i}"
	im1 = Image.open(f"{output_dir}/{tsne_mine}.png")
	im2 = Image.open(f"{output_dir}/{tsne_mnist}.png")

	im3 =Image.blend(im1,im2,0.3)
	im3.save(f"{main}/blend_{i}.png")
