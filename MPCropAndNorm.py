import cv2
import mediapipe as mp
import numpy as np
import copy

class MPCropAndNorm:
    def __init__(self, min_detection_confidence=0.7, static_image_mode=True, max_num_faces=1, minimal_scale_ratio=0.75):
        # Initialize face detection and face mesh
        mp_face_detection = mp.solutions.face_detection
        self.__face_detection = mp_face_detection.FaceDetection(min_detection_confidence=min_detection_confidence)
        
        mp_face_mesh = mp.solutions.face_mesh
        self.__face_mesh = mp_face_mesh.FaceMesh(static_image_mode=static_image_mode, max_num_faces=max_num_faces)
        
        self.__minimal_scale_ratio = minimal_scale_ratio
        self.__avg_y_on_1024 = 0.6076579708466568 * 1024
        self.__standard_dxdy = 105.64138908808019 * 1024

    def detect_face(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.__face_detection.process(img_rgb)
        faces = []
        if results.detections:
            for detection in results.detections:
                box = detection.location_data.relative_bounding_box
                x, y, w, h = box.xmin, box.ymin, box.width, box.height
                faces.append((int(x*img.shape[1]), int(y*img.shape[0]), int(w*img.shape[1]), int(h*img.shape[0])))
        return faces

    def __align_face_original(self, img, face):
        x, y, w, h = face
        img_rgb = cv2.cvtColor(img[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
        results = self.__face_mesh.process(img_rgb)
        facial_pose = []
        if results.multi_face_landmarks:
            for landmark in results.multi_face_landmarks[0].landmark:
                facial_pose.append(((x + w * landmark.x), (y + h * landmark.y)))
        return facial_pose

    def __count_pose(self, facial_pose):
        xs = [x for x, y in facial_pose]
        ys = [y for x, y in facial_pose]
        avg_x = sum(xs) / len(xs)
        avg_y = sum(ys) / len(ys)
        sum_dxdy = sum(abs(x - avg_x) + abs(y - avg_y) for x, y in facial_pose)
        return avg_x, avg_y, sum_dxdy

    def __safe_crop(self, img, x0, x1, y0, y1, if_padding=True):
        h, w = img.shape[:2]
        # 初始化填充的边界
        pad_left = -min(0, x0)
        pad_right = max(x1 - w, 0)
        pad_top = -min(0, y0)
        pad_bottom = max(y1 - h, 0)

        # 更新坐标以保证它们在图像范围内
        x0 = max(0, x0)
        x1 = min(w, x1)
        y0 = max(0, y0)
        y1 = min(h, y1)

        cropped_img = img[y0:y1, x0:x1]

        # 如果需要填充
        if if_padding and (pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0):
            # print("padding", pad_top, pad_bottom)
            if img.ndim == 3:  # 彩色图像
                pad_width = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
            else:  # 灰度图像
                pad_width = ((pad_top, pad_bottom), (pad_left, pad_right))

            # 选择填充模式，这里使用边缘值填充
            mode = 'edge'
            cropped_img = np.pad(cropped_img, pad_width=pad_width, mode=mode)

        return cropped_img
    
    def crop_and_norm_with_face( self, img , faces ):
        ans = []
        for face in faces:
            facial_pose = self.__align_face_original(img, face)
            if facial_pose:
                avg_x, avg_y, sum_dxdy = self.__count_pose(facial_pose)
                
                scale = sum_dxdy / self.__standard_dxdy
                x0 = round(avg_x - 511.5 * scale)
                x1 = round(avg_x + 511.5 * scale)
                y0 = round(avg_y - self.__avg_y_on_1024 * scale)
                y1 = round(avg_y + (1024 - self.__avg_y_on_1024) * scale)
                # print(avg_x,scale,x0,x1,y0,y1)
                sub_img = self.__safe_crop(img, x0, x1, y0, y1, True)
                sub_img = copy.deepcopy(sub_img)
                ans.append( sub_img )
        return ans


    def crop_and_norm(self, img ):
        faces = self.detect_face(img)
        h, w = img.shape[:2]
        return self.crop_and_norm_with_face( img , faces )

# Example usage:
# detector = MPCropAndNorm()
# image = cv2.imread('example.png')
# faces = detector.crop_and_norm(image, 0)
