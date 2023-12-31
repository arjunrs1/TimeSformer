import json

label_map = {'Novice': 0, 'Early Expert': 3, 'Intermediate Expert': 2, 'Late Expert': 1}
test_annotations_file = "/vision/asomaya1/ego_exo/data/demonstrator_arxiv23_test.json"

with open(test_annotations_file, 'r') as file:
    data = json.load(file)


video_ids = list()
scenario_names = list()
gt_annotations = list()

for item in data:
    # Extract video id from one of the video paths
    video_path = item["video_paths"]["ego"]  # Using "ego" path as an example
    video_id = video_path.split('/')[1]
    video_ids.append(video_id)

    # Add scenario name
    scenario_names.append(item["scenario_name"])

    # Add proficiency score
    proficiency_score = label_map[item["proficiency_score"]]
    gt_annotations.append(proficiency_score)

# Create the resulting JSON structure
result = {
    "videos": list(video_ids),
    "scenario_names": list(scenario_names),
    "gt_annotations": list(gt_annotations)
}

output_file_path = "test_annotations.json"
with open(output_file_path, 'w') as file:
    json.dump(result, file)