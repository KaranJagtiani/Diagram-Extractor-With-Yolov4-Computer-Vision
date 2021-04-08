import glob
import os
import detector as det

image_files = glob.glob("Testing/*png") \
+ glob.glob("Testing/*jpg") \
+ glob.glob("Testing/*jpeg")

print("Image Files:", image_files)

i = 1
for image in image_files:
    print(f"Processing Image: {image}.")
    l1 = None
    try:
        l1 = det.objDetector(image, i)
    except:
        print("Could not detect")
    i+=1
    print(l1)
    print(f"Done for: {image}")
