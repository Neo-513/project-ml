import os
import pickle


def read(file_path):
	extension = os.path.splitext(file_path)[-1]
	if extension == ".pkl":
		with open(file_path, mode="rb") as file:
			datas = pickle.load(file)
	else:
		with open(file_path, mode="r", encoding="utf-8") as file:
			datas = file.read()
	print(f"[READ] {file_path}")
	return datas


def write(file_path, datas):
	extension = os.path.splitext(file_path)[-1]
	if extension == ".pkl":
		with open(file_path, mode="wb") as file:
			pickle.dump(datas, file)
	else:
		with open(file_path, mode="w", encoding="utf-8") as file:
			file.write(datas)
	print(f"[WRITE] {file_path}")
