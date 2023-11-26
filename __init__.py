import importlib
import os
import requests
from folder_paths import models_dir

node_list = ["safety_checker"]

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

def download_file(url, destination):
    if os.path.exists(destination):
        return False

    with requests.get(url, stream=True) as response:
        total_length = response.headers.get('content-length')

        if total_length is None:
            open(destination, 'wb').write(response.content)
        else:
            dl = 0
            total_length = int(total_length)
            with open(destination, 'wb') as file:
                for data in response.iter_content(chunk_size=4096):
                    dl += len(data)
                    file.write(data)
                    done = int(50 * dl / total_length)
                    print(f"\rSafety Checker: Downloading {os.path.basename(destination)}: [{'=' * done}{' ' * (50-done)}] {dl}/{total_length} bytes", end='')
    return True

def remove_lines_from_file(file_path, line_numbers):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    with open(file_path, 'w') as file:
        for i, line in enumerate(lines, start=1):
            if i not in line_numbers:
                file.write(line)

def setup_safety_checker():
    safety_checker_dir = os.path.join(models_dir, "safety_checker")

    if not os.path.exists(safety_checker_dir):
        os.makedirs(safety_checker_dir)
        print("Safety Checker: Created directory: " + safety_checker_dir)

    files_to_download = {
        "pytorch_model.bin": "https://huggingface.co/CompVis/stable-diffusion-safety-checker/resolve/main/pytorch_model.bin",
        "config.json": "https://huggingface.co/CompVis/stable-diffusion-safety-checker/resolve/main/config.json",
        "preprocessor_config.json": "https://huggingface.co/CompVis/stable-diffusion-safety-checker/resolve/main/preprocessor_config.json"
    }

    for filename, url in files_to_download.items():
        destination_path = os.path.join(safety_checker_dir, filename)
        if download_file(url, destination_path) and filename == "config.json":
            remove_lines_from_file(destination_path, [32, 33, 34, 35])

for module_name in node_list:
    imported_module = importlib.import_module(".{}".format(module_name), __name__)
    NODE_CLASS_MAPPINGS = {**NODE_CLASS_MAPPINGS, **imported_module.NODE_CLASS_MAPPINGS}
    NODE_DISPLAY_NAME_MAPPINGS = {**NODE_DISPLAY_NAME_MAPPINGS, **imported_module.NODE_DISPLAY_NAME_MAPPINGS}

setup_safety_checker()

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
