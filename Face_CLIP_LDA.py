import cv2
import pickle
from MPCropAndNorm import MPCropAndNorm
from CLIPExtractor import CLIPExtractor
from huggingface_hub import hf_hub_download

class Face_CLIP_LDA:
    def __init__(self, lda_model_path = None):
        # 加载LDA模型
        if lda_model_path == None:
            lda_model_path = hf_hub_download(repo_id="silk-road/simple-face-recognition", filename="lda_openai_clip_model.pkl")
        
        with open(lda_model_path, 'rb') as f:
            self.lda_model = pickle.load(f)

        self.detector = MPCropAndNorm()
        self.clip_extractor = CLIPExtractor()

    def project_to_lda(self, feature):
        # 将特征投影到LDA空间
        return self.lda_model.transform([feature])

    def extract(self, file_names):
        # 从文件中提取特征
        features = self.clip_extractor.extract(file_names)
        # 将特征投影到LDA空间
        projected_features = [self.project_to_lda(feature)[0] for feature in features]
        return projected_features

    def detect_and_extract(self, image):
        # 检测人脸并裁剪
        faces = self.detector.crop_and_norm(image)
        face_features = []

        # 保存裁剪后的人脸到临时文件，以便进行特征提取
        temp_file_names = []
        for i, face in enumerate(faces):
            temp_file_name = f"temp_face_{i}.jpg"
            cv2.imwrite(temp_file_name, face)
            temp_file_names.append(temp_file_name)

        # 提取特征
        features = self.clip_extractor.extract(temp_file_names)
        for feature in features:
            # 将特征投影到LDA空间
            projected_feature = self.project_to_lda(feature)[0]
            face_features.append(projected_feature)

        # 返回检测到的人脸和对应的特征
        return faces, face_features

# 示例用法
# from Face_CLIP_LDA import Face_CLIP_LDA
# 对于已经crop过的图片
# face_feature_extractor = Face_CLIP_LDA()
# file_names = ["cropped_face.jpg"]
# face_features = face_feature_extractor.extract(file_names)
# 或者，对于一张新图片
# image = cv2.imread('example.png')
# faces, face_features = face_feature_extractor.detect_and_extract(image)
