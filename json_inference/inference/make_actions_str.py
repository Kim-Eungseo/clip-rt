import json

def convert_list_values_to_str(input_json_path, output_json_path):
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for key, value in data.items():
        if isinstance(value, list):
            data[key] = str(value)  

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    input_path = "/home/ngseo/libero/clip-rt/json_inference/inference/libero_spatial_command_to_action_kmeans_32_nonzero.json"
    output_path = "/home/ngseo/libero/clip-rt/json_inference/inference/libero_spatial_command_to_action_kmeans_32_nonzero.json"
    convert_list_values_to_str(input_path, output_path)
