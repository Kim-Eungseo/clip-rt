import json

def convert_list_values_to_str(input_json_path, output_json_path):
    # 1) JSON 로드
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 2) 각 key-value 쌍 확인
    #    value가 list이면 str(value)로 변환. 예: [1, 2, 3] -> "[1, 2, 3]"
    for key, value in data.items():
        if isinstance(value, list):
            data[key] = str(value)  # 리스트 전체를 문자열로 바꾸기

    # 3) 변환된 데이터를 JSON 파일로 저장
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    input_path = "/home/ngseo/libero/clip-rt/json_inference/inference/libero_spatial_command_to_action_kmeans_32.json"
    output_path = "/home/ngseo/libero/clip-rt/json_inference/inference/libero_spatial_command_to_action_kmeans_32.json"
    convert_list_values_to_str(input_path, output_path)
    print(f"'{input_path}'의 리스트 타입 value를 문자열로 변환하여 '{output_path}'에 저장했습니다.")
