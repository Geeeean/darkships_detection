import os
import shutil


class Utils:
    @staticmethod
    def create_empty_folder(folder_path):
        if os.path.exists(folder_path):
            if os.path.isdir(folder_path):
                for item in os.listdir(folder_path):
                    item_path = os.path.join(folder_path, item)
                    if os.path.isfile(item_path):
                        os.remove(item_path)
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
            else:
                os.remove(folder_path)
                os.makedirs(folder_path)
        else:
            os.makedirs(folder_path)

    @staticmethod
    def _ls(path: str, contains: str | None):
        if not os.path.exists(path) or not os.path.isdir(path):
            print(f"Error: '{path}' doesn't exist or isn't a folder.")
            return []

        files = []

        for filename in os.listdir(path):
            if contains is None or contains in filename:
                file_path = os.path.join(path, filename)

                if os.path.isfile(file_path):
                    files.append(file_path)

        files.sort()
        return files
