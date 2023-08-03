import os

folder_path = '/home/jovyan/work/datasets/satellite3/all_train_img'
outlier_file = '/home/jovyan/work/gitsubmit/pre_processing/outlier.txt'

with open(outlier_file, 'r') as f:
    outlier_words = f.read().splitlines()

# 폴더 내 파일 목록 얻기
file_list = os.listdir(folder_path)

# 파일 목록 순회하며 outlier 단어로 끝나는 파일 삭제
cnt = 0

for file_name in file_list:
    for outlier_word in outlier_words:
        if file_name.endswith(outlier_word + ".png"):
            file_path = os.path.join(folder_path, file_name)
            os.remove(file_path)
            cnt = cnt + 1
            print(f"{file_name} 삭제 완료")

print("삭제 작업 완료")
print(cnt)