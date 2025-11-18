import cv2
import numpy as np

# Load image
img = cv2.imread('temp_orthofoto.png')
if img is None:
    print("❌ No image found!")
    exit()

print("📐 Image loaded:", img.shape)

# Convert to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)

print("\n📊 Image Statistics:")
print(f"V (brightness) - Min: {v.min()}, Max: {v.max()}, Mean: {v.mean():.1f}")
print(f"S (saturation) - Min: {s.min()}, Max: {s.max()}, Mean: {s.mean():.1f}")
print(f"H (hue) - Min: {h.min()}, Max: {h.max()}, Mean: {h.mean():.1f}")

# Test different darkness thresholds
print("\n🔍 Testing darkness thresholds:")
for threshold in [60, 80, 100, 120, 140]:
    dark_pixels = np.sum(v < threshold)
    percent = (dark_pixels / v.size) * 100
    print(f"   V < {threshold}: {dark_pixels:,} pixels ({percent:.2f}%)")

# Show different masks
dark_mask = (v < 100).astype(np.uint8) * 255
blue_mask = ((h >= 90) & (h <= 130) & (v < 120)).astype(np.uint8) * 255
black_mask = ((s < 50) & (v < 90)).astype(np.uint8) * 255

# Combine
combined = cv2.bitwise_or(dark_mask, blue_mask)
combined = cv2.bitwise_or(combined, black_mask)

# Clean up
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)

# Find contours
contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"\n🔍 Found {len(contours)} contours")

# Filter by size
min_area = 100
max_area = 50000
filtered = [c for c in contours if min_area <= cv2.contourArea(c) <= max_area]
print(f"   After size filter ({min_area}-{max_area}): {len(filtered)} contours")

# Draw on image
result = img.copy()
for contour in filtered:
    area = cv2.contourArea(contour)
    x, y, w, h = cv2.boundingRect(contour)
    
    # Get average darkness
    roi_v = v[y:y+h, x:x+w]
    avg_v = np.mean(roi_v)
    
    cv2.drawContours(result, [contour], -1, (0, 255, 255), 2)
    cv2.putText(result, f"{avg_v:.0f}", (x, y-5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

print("\n📊 Top 10 darkest regions:")
darkest = []
for contour in filtered:
    x, y, w, h = cv2.boundingRect(contour)
    roi_v = v[y:y+h, x:x+w]
    avg_v = np.mean(roi_v)
    darkest.append((avg_v, x, y, w, h, cv2.contourArea(contour)))

darkest.sort()
for i, (avg_v, x, y, w, h, area) in enumerate(darkest[:10]):
    print(f"   {i+1}. Position ({x},{y}) Size {w}x{h} Area {area:.0f} Brightness {avg_v:.0f}")

# Save visualization
cv2.imwrite('debug_dark_mask.png', dark_mask)
cv2.imwrite('debug_blue_mask.png', blue_mask)
cv2.imwrite('debug_black_mask.png', black_mask)
cv2.imwrite('debug_combined_mask.png', combined)
cv2.imwrite('debug_cleaned_mask.png', cleaned)
cv2.imwrite('debug_detections.png', result)

print("\n💾 Saved debug images:")
print("   - debug_dark_mask.png (V < 100)")
print("   - debug_blue_mask.png (blue dark areas)")
print("   - debug_black_mask.png (black areas)")
print("   - debug_combined_mask.png (all masks combined)")
print("   - debug_cleaned_mask.png (after morphology)")
print("   - debug_detections.png (detected regions with brightness values)")

print("\n✅ Check these images to see what's being detected!")
print("   If solar panels aren't showing up, they might not be dark enough.")
print("   Try adjusting thresholds or using the ML training approach.")
