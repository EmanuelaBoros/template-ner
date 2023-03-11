import pandas as pd
import logging
from seq2seq_model import Seq2SeqModel
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_file",
        type=str,
    )
    parser.add_argument(
        "--dev_file",
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
    )


    args, _ = parser.parse_known_args()

train_df = pd.read_csv(args.train_file)
# train_df = pd.DataFrame(train_data, columns=["input_text", "target_text"])
print(train_df.head())
eval_df = pd.read_csv(args.dev_file)
# eval_df = pd.DataFrame(eval_data, columns=["input_text", "target_text"])


model_args = {
    "reprocess_input_data": True,
    "overwrite_output_dir": True,
    "max_seq_length": 50,
    "train_batch_size": 16,
    "num_train_epochs": 20,
    "save_eval_checkpoints": False,
    "save_model_every_epoch": False,
    "evaluate_during_training": True,
    "evaluate_generated_text": True,
    "evaluate_during_training_verbose": True,
    "use_multiprocessing": False,
    "max_length": 25,
    "manual_seed": 4,
    "save_steps": 11898,
    "gradient_accumulation_steps": 1,
    "output_dir": "./exp/template",
    "n_gpu": 2
}


# Initialize model
model = Seq2SeqModel(
    encoder_decoder_type="mbart",
    encoder_decoder_name="facebook/mbart-large-50",
    args=model_args,
    # use_cuda=False,
)


# Train the model
model.train_model(train_df, eval_data=eval_df)

# Evaluate the model
results = model.eval_model(eval_df)

# Use the model for prediction

print(model.predict(["Japan began the defence of their Asian Cup title with a lucky 2-1 win against Syria in a Group C championship match on Friday."]))
