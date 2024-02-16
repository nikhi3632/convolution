import cv2, os

def check_image_difference(image_path1, image_path2, threshold):
    image1 = cv2.imread(image_path1)
    image2 = cv2.imread(image_path2)
    if image1.shape != image2.shape:
        return True
    diff = cv2.absdiff(image1, image2)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresholded_diff = cv2.threshold(gray_diff, threshold, 255, cv2.THRESH_BINARY)
    as_difference = cv2.countNonZero(thresholded_diff) > 0
    return as_difference

image_path1 = os.getcwd() + "/conv2_output_v1.jpeg"
image_path2 = os.getcwd() + "/artifacts/conv2_output_v2.jpeg"
threshold = 2

if check_image_difference(image_path1, image_path2, threshold):
    print("The images have differences above the threshold.")
else:
    print(f"The images are identical or have differences below the threshold={threshold} .")
