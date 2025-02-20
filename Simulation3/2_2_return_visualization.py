### Select the sample sizes with which we MC-approximated the return distribution (marginal).
display_samplesize = 100000 

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image_paths = ['materials/setting' + str(i) + '_N' + str(display_samplesize) + '.jpg' for i in (1,2,3)]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))  

for i, image_path in enumerate(image_paths):
    img = mpimg.imread(image_path)
    axes[i].imshow(img)
    axes[i].axis('off')  

plt.tight_layout()
output_path = 'materials/settings123_return.jpg' 
plt.savefig(output_path, dpi=300, bbox_inches='tight')  
plt.close(fig)  

