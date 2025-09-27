import cv2

# Check version
print(cv2.__version__)

# Try to access SURF
try:
    surf = cv2.xfeatures2d.SURF_create()
    print("SURF is available ✅")
except AttributeError:
    print("SURF is NOT available ❌")
