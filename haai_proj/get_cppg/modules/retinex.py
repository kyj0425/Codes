import numpy as np
import cv2


import numpy as np
import cv2

class RetinexEnhancer:
    def __init__(self, sigma=30, low_percent=1, high_percent=99):
        self.sigma = sigma
        self.low_percent = low_percent
        self.high_percent = high_percent

    def fastRetinex(self, img):
        img = np.float32(img) + 1.0
        blur = cv2.GaussianBlur(img, (0, 0), self.sigma)
        retinex = np.log(img) - np.log(blur)

        # 정규화
        retinex = self.percentile_clip(retinex, self.low_percent, self.high_percent)
        retinex = np.uint8((retinex - retinex.min()) / (retinex.max() - retinex.min()) * 255)

        return retinex

    def percentile_clip(self, img, low_percent, high_percent):
        """
        가장 빠른 클리핑 함수 (각 채널별로 percentile로 min/max 결정)
        """
        out = np.zeros_like(img)
        for i in range(img.shape[2]):
            low_val = np.percentile(img[:, :, i], low_percent)
            high_val = np.percentile(img[:, :, i], high_percent)
            out[:, :, i] = np.clip(img[:, :, i], low_val, high_val)
        return out

