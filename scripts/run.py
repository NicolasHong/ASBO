import papermill as pm
from pathlib import Path
BASE_DIR = Path().absolute()/'scripts'

print(BASE_DIR)
# 定义你的 notebook 文件名
notebooks = [
    BASE_DIR /'Case 1.ipynb',
    BASE_DIR /'Case 2.ipynb',
    BASE_DIR / 'Case 3.ipynb'
]
# notebooks = ['Case 1.ipynb', 'Case 2.ipynb', 'Case 3.ipynb']

for notebook in notebooks:
    pm.execute_notebook(notebook, notebook)  # 将结果保存到原文件