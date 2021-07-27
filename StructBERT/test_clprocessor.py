from run_classifier_multi_task import DataProcessor, CommonLitProcessor

data_dir="/Users/bayartsogtyadamsuren/Projects/kaggle/AliceMind/StructBERT/data/commonlit/fold_0"
data = DataProcessor._read_tsv_pandas(f"{data_dir}/valid.tsv")
print(data.shape)
print(data.head())
examples = CommonLitProcessor().get_dev_examples(data_dir)
for example in examples[:3]:
    print(example)
    print(example.guid)
    print(example.text_a)
    print(example.text_b)
    print(example.label)