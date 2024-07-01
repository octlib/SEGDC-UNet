import os
import numpy as np
from PIL import Image
import imgaug.augmenters as iaa

# 定义增强方法和次数
augmentations = iaa.Sequential([
    iaa.Affine(rotate=(-20, 20)),
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    iaa.Multiply((0.8, 1.2)),  # 调整亮度
    iaa.GaussianBlur(sigma=(0, 3.0))
], random_order=True)  # 将增强方法以随机顺序应用，以生成更多变化的图像

# 输入文件夹路径
input_image_folder = "E:/data/emps/images"  # 输入图像文件夹
input_mask_folder = "E:/data/emps/segmaps"  # 输入掩码图像文件夹

# 输出文件夹路径
output_image_folder = "E:/data/mix/images"  # 输出图像文件夹
output_mask_folder = "E:/data/mix/segmaps"  # 输出掩码图像文件夹

# 确保输出文件夹存在
if not os.path.exists(output_image_folder):
    os.makedirs(output_image_folder)
if not os.path.exists(output_mask_folder):
    os.makedirs(output_mask_folder)

# 加载所有输入图像
input_image_files = os.listdir(input_image_folder)

# 定义每张图像应用增强的次数
augmentation_count_per_image = 10  # 每张图像增强 10 次

# 生成增强图像和掩码
for filename in input_image_files:
    # 读取图像
    img_path = os.path.join(input_image_folder, filename)
    mask_path = os.path.join(input_mask_folder, filename)
    img = Image.open(img_path)
    mask = Image.open(mask_path)

    # 将图像转换为数组
    img_array = np.array(img)   # (w,h,3)
    mask_array = np.array(mask)     #(w,h)
    mask_array = np.expand_dims(mask_array, axis=2).astype(np.uint8) #(w,h,1)

    # 为了将 image 和 mask 做同样的变换，将两者链接在一起
    img_mask_array = np.concatenate((img_array, mask_array), axis=2)    # (w,h,4)

    # 应用增强
    augmented_images_masks = [augmentations(image=img_mask_array) for _ in range(augmentation_count_per_image)]

    # 保存增强后的图像
    for idx, augmented_img_mask in enumerate(augmented_images_masks):
        # 创建保存路径
        output_filename = f"{os.path.splitext(filename)[0]}_aug_{idx}.png"
        output_img_path = os.path.join(output_image_folder, output_filename)
        output_mask_path = os.path.join(output_mask_folder, output_filename)

        # 将image和mask分开，保存图像
        augmented_img = augmented_img_mask[:, :, :3]
        augmented_mask = augmented_img_mask[:, :, 3:]
        augmented_mask = np.squeeze(augmented_mask)

        # 检查最大值是否为零
        #max_value = np.max(augmented_mask)
        #if max_value > 0:
            #augmented_mask = augmented_mask * (250 // max_value)  # 之前的mask太暗了，变亮一些

        Image.fromarray(augmented_img).save(output_img_path)
        Image.fromarray(augmented_mask).save(output_mask_path)

print("图像增强完成！")
