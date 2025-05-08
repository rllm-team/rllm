from sentence_transformers import SentenceTransformer
from rllm.types import ColType
from rllm.data import TableData, TextEmbedderConfig
import torch
import pandas as pd


csv_content = """Column1,Column2,Column3,Column4,Column5,Column6
Value1,Value2,22,1,"hello",Value6
Value7,Value8,355,2,"this is",Value12
Value13,Value14,67,35,"a test",Value18
Value19,Value20,88,64,"thanks for your attention!",Value24
"""
with open("test.csv", "w", encoding="utf-8", newline="") as f:
    f.write(csv_content)
df = pd.read_csv("./test.csv")


class embedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def __call__(self, texts):
        return torch.tensor(self.model.encode(texts))


col_types = {
    "Column1": ColType.CATEGORICAL,
    "Column2": ColType.CATEGORICAL,
    "Column3": ColType.NUMERICAL,
    "Column4": ColType.NUMERICAL,
    "Column5": ColType.TEXT,
    "Column6": ColType.TEXT,
}
cfg = TextEmbedderConfig(text_embedder=embedder(), batch_size=8)
data = TableData(
    df=df, col_types=col_types, target_col="Survived", text_embedder_config=cfg
)
print(data.feat_dict)
