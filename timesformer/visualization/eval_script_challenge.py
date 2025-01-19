import argparse
import json
import numpy as np
import tqdm
from collections import defaultdict

def validate_model_predictions(model_predictions, test_annotations):
    # Validate model predictions global file structure:
    assert type(model_predictions) == dict
    for key in ["videos", "predictions"]:
        assert key in model_predictions.keys()

    assert model_predictions["videos"] == test_annotations["videos"]
    
    # Sanity check the test annotation file:
    assert type(test_annotations["gt_annotations"]) == list
    assert type(test_annotations["scenario_names"]) == list
    assert len(test_annotations["gt_annotations"]) == len(test_annotations["videos"])
    assert len(test_annotations["scenario_names"]) == len(test_annotations["videos"])

    # Validate ego and exo model predictions structure:
    assert len(model_predictions["predictions"]) == len(test_annotations["videos"])
    for prediction in model_predictions["predictions"]:
        assert prediction == int(prediction)

def softmax(x):
    """Compute softmax values for each set of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def evaluate_performance(gt_annotations, model_predictions, model_type='combined'):
    correct_predictions = 0
    total_predictions = len(gt_annotations)

    for gt_label, prediction in zip(
        gt_annotations,
        model_predictions["predictions"]
    ):
        if prediction == gt_label:
            correct_predictions += 1

    accuracy = correct_predictions / total_predictions
    return accuracy

def evaluate_performance_by_scenario(gt_annotations, model_predictions, scenarios):
    scenario_accuracies = defaultdict(lambda: {'correct': 0, 'total': 0})

    for gt_label, prediction, scenario in zip(
        gt_annotations["gt_annotations"],
        model_predictions["predictions"],
        scenarios
    ):
        if prediction == gt_label:
            scenario_accuracies[scenario]['correct'] += 1
        scenario_accuracies[scenario]['total'] += 1

    return weighted_top1_accuracy(scenario_accuracies)

def weighted_top1_accuracy(results):
    total_correct = 0
    total_samples = 0
    
    for class_label, data in results.items():
        total_correct += data['correct']
        total_samples += data['total']
    
    if total_samples == 0:
        return 0
    return (total_correct / total_samples) * 100


def evaluate(test_annotation_file, user_annotation_file):
    print("Starting Evaluation.....")

    with open(test_annotation_file, "r") as fp:
        gt_annotations = json.load(fp)
    with open(user_annotation_file, "r") as fp:
        model_predictions = json.load(fp)

    # Validate model predictions:
    validate_model_predictions(model_predictions, gt_annotations)

    top1_accuracy = evaluate_performance(gt_annotations["gt_annotations"], model_predictions)
    scenario_accuracies = evaluate_performance_by_scenario(gt_annotations, model_predictions, gt_annotations["scenario_names"])

    output = {}
    output['result'] = [
        {
            'test_split': {
                'accuracy_top1': top1_accuracy
            }
        }
    ]
    print(output)
    return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-annotation-file", required=True, type=str)
    parser.add_argument("--user-annotation-file", required=True, type=str)

    args = parser.parse_args()

    evaluate(args.test_annotation_file, args.user_annotation_file)