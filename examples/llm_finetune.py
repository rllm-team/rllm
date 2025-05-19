import sys

sys.path.append("./")
sys.path.append("../")
from rllm.llm import FinetuneConfig, Seq2SeqFinetuner


if __name__ == "__main__":
    # Example usage
    cfg = FinetuneConfig(
        csv_dataset="Your CSV file path here",
        target_column="Your target column name here",
        task_info_path="Your task info file path(txt format) here",
    )
    finetuner = Seq2SeqFinetuner(cfg)
    finetuner.train()
