from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from dataset import DatasetForTrainer

class Model:
    def tokenize(self, train_texts, test_texts):
        self.tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

        self.train_encodings = self.tokenizer(train_texts, truncation=True, padding=True)
        self.test_encodings = self.tokenizer(test_texts, truncation=True, padding=True)

    def prepare_for_training_w_torch(self, train_labels, test_labels):
        self.train_dataset = DatasetForTrainer(self.train_encodings, train_labels)
        self.test_dataset = DatasetForTrainer(self.test_encodings, test_labels)

    def train(self):
        self.model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=2)
        
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        self.training_args = TrainingArguments(
            output_dir="distilbert-fake-real-news",
            learning_rate=2e-5,
            eval_strategy="epoch",
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=2,
            push_to_hub=False,
        )

        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
        )

        self.trainer.train()