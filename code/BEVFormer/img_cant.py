import cv2
import numpy as np

# 读取图片
img1 = cv2.imread('outputs/08e76760a8c64a92a86686baf68f6aff_camera.png')
img2 = cv2.imread('outputs/08e76760a8c64a92a86686baf68f6aff_bev.png')

# 将img2的高度缩小为img1的一半
new_height = img1.shape[0] // 2
new_width = int(img2.shape[1] * (new_height / img2.shape[0]))
img2_resized = cv2.resize(img2, (new_width, new_height))

# 创建一个新的图像，宽度为img1的宽度加上img2的宽度，高度为img1的高度
new_img = np.zeros((img1.shape[0], img1.shape[1] + img2_resized.shape[1], 3), dtype=np.uint8)

# 将img1和img2拼接到新的图像上
new_img[:, :img1.shape[1]] = img1
new_img[:img2_resized.shape[0], img1.shape[1]:] = img2_resized

cv2.imwrite(f'output_test.jpg', new_img)
# 显示新的图像
#cv2.imshow('New Image', new_img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
