
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

class CLIPExtractor:
    def __init__(self, model_name="openai/clip-vit-base-patch16", processor_name="openai/clip-vit-base-patch16"):
        # Initialize the model and processor with default or specified values
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(processor_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def extract(self, file_names, batch_size=16):
        num_images = len(file_names)
        all_features = []

        for start_idx in tqdm(range(0, num_images, batch_size)):
            batch_files = file_names[start_idx:start_idx + batch_size]
            images = [Image.open(file_name).convert("RGB") for file_name in batch_files]
            inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)

            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)

            all_features.extend(outputs.cpu().numpy())

        return all_features

    def visualize(self, features, save_name):
        features_np = np.array(features)
        plt.figure(figsize=(10, 8))
        plt.imshow(features_np, aspect='auto', cmap='viridis')
        plt.colorbar()
        plt.title('Feature Heatmap')
        plt.xlabel('Feature Index')
        plt.ylabel('Image Index')
        plt.savefig(save_namenump)
        plt.close()

if __name__ == "__main__":
    extractor = CLIPExtractor()
    import os
    # 遍历解压后的文件夹，获取所有.jpg文件名
    jpg_files = []
    folder_path = "./output/temp_crop"
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".jpg"):
                jpg_files.append(os.path.join(root, file))
    file_names = jpg_files
    features = extractor.extract(file_names)
    extractor.visualize(features, 'output/feature_heatmap.png')

# Example usage:
# extractor = CLIPExtractor()
# file_names = ['path/to/your/image1.jpg', 'path/to/your/image2.jpg'] # Replace these paths with your actual file paths
# features = extractor.extract(file_names)
# extractor.visualize(features, 'heatmap.png')



# 我需要把下面代码重构成 CLIPExtractor类

# 已知下面这段代码

# ```python
# from transformers import CLIPProcessor, CLIPModel
# from PIL import Image
# import torch
# from tqdm import tqdm

# # Define the batch size
# batch_size = 16

# model_name = "Green-Sky/FaRL-Base-Patch16-LAIONFace20M-ep64"
# processor_name = "openai/clip-vit-base-patch16"

# # Initialize the model and processor
# model = CLIPModel.from_pretrained(model_name)
# processor = CLIPProcessor.from_pretrained(processor_name)

# # model = CLIPModel.from_pretrained("h94/IP-Adapter")
# # processor = CLIPProcessor.from_pretrained("h94/IP-Adapter")

# device = "cuda" if torch.cuda.is_available() else "cpu"
# model.to(device)

# # Function to process images in batches and extract features
# def batch_process_images(file_names, batch_size, model, processor):
#     num_images = len(file_names)
#     all_features = []

#     for start_idx in tqdm(range(0, num_images, batch_size)):
#         batch_files = file_names[start_idx:start_idx + batch_size]
#         images = [Image.open(file_name).convert("RGB") for file_name in batch_files]

#         inputs = processor(images=images, return_tensors="pt", padding=True).to(device)

#         with torch.no_grad():
#             outputs = model.get_image_features(**inputs)

#         all_features.extend(outputs.cpu().numpy())

#     return all_features

# # Example usage
# batch_size = 16
# file_names = jpg_files
# features = batch_process_images(file_names, batch_size, model, processor)
# ```

# 能够正常运行

# 以及可视化代码

# ```python
# import numpy as np
# import matplotlib.pyplot as plt

# # 假设`features`是一个二维数组，其中每一行是一个图像的特征向量
# # 例如: features = np.random.rand(5, 2048) # 使用随机数据作为示例

# # 将features转换为NumPy数组，以便更容易地处理
# features_np = np.array(features)

# # 创建热图
# plt.figure(figsize=(10, 8))
# plt.imshow(features_np, aspect='auto', cmap='viridis')
# plt.colorbar()
# plt.title('Feature Heatmap')
# plt.xlabel('Feature Index')
# plt.ylabel('Image Index')
# plt.show()
# ```

# 我希望实现一个CLIPExtractor类

# 这个类可以默认初始化（无参数）

# extractor = CLIPExtractor()

# 也可以指定model_name和process_name进行初始化

# 然后使用features = extractor.extract( file_names )

# 进行特征的抽取

# 以及一个visualize函数extractor.visualize( features, save_name )

# 可以将可视化的热图保存到save_name中
