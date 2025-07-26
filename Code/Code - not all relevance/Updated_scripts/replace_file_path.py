import json
import os

# # Load annotations.json
# with open("annotations.json", "r", encoding="utf-8") as f:
#     annotations = json.load(f)
#
# # Check dataset stats
# print(f"Total dataset size: {len(annotations)}")
#
# # Verify a few samples
# for sample in annotations[:5]:  # Check first 5 entries
#     print(f"Image: {sample['image']}\nText: {sample['text']}\n---")
#
# # Ensure file paths use forward slashes for compatibility
# for entry in annotations:
#     entry["image"] = entry["image"].replace("\\", "/")
#
# # Save the fixed annotations.json
# with open("annotations_fixed.json", "w", encoding="utf-8") as f:
#     json.dump(annotations, f, indent=4)
#
# with open("annotations_fixed.json", "r", encoding="utf-8") as file:
#     annotations_fixed = json.load(file)
#
# print("✅ Dataset verified and saved as annotations_fixed.json")
# print(f"fixed annotations size is: {len(annotations_fixed)}")


# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------

# Load original annotations
with open("annotations_fixed.json", "r") as f:
    annotations = json.load(f)

# Define new base path
drive_base_path = "/content/drive/MyDrive/ColabNotebooks/images"

# Update image paths
for entry in annotations:
    relative_path = entry["image"].replace("./images", "")  # e.g. "/credit_card/xyz.png"
    entry["image"] = drive_base_path + relative_path

# Save updated annotations
with open("annotations_fixed_drive.json", "w") as f:
    json.dump(annotations, f, indent=4)