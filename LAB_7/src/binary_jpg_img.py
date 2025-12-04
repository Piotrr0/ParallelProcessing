from PIL import Image
import numpy as np

image_path = 'Gwny_budynek_PK.jpg'
img = Image.open(image_path).convert('RGB')
img_np = np.array(img)
h, w, c = img_np.shape
print("Shape (h,w,c):", img_np.shape)

img_np.astype(np.uint8).tofile("image.raw")

out_np = np.fromfile("output.raw", dtype=np.uint8).reshape((h, w, c))
out_img = Image.fromarray(out_np)
out_img.save("processed.jpg")