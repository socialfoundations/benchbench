from datasets import load_dataset

dataset = load_dataset("gsm8k", name="main", split="test")
print("gsm8k")
print(len(set([eval(i.split("#### ")[-1]) for i in dataset["answer"]])), len(dataset))
