import pandas as pd
import logging
from seq2seq_model import Seq2SeqModel
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)
import argparse
import re

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
        "--test_file",
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
    )
    parser.add_argument(
        "--do_train",
        action="store_const",
    )
    parser.add_argument(
        "--do_test",
        action="store_const",
    )
    parser.add_argument(
        "--do_eval",
        action="store_const",
    )
    parser.add_argument("--reprocess_input_data", type=bool, default=True, help="Reprocess input data")
    parser.add_argument("--overwrite_output_dir", type=bool, default=True, help="Overwrite output directory")
    parser.add_argument("--max_seq_length", type=int, default=50, help="Max sequence length")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--num_train_epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--save_eval_checkpoints", type=bool, default=False, help="Save evaluation checkpoints")
    parser.add_argument("--save_model_every_epoch", type=bool, default=False, help="Save model every epoch")
    parser.add_argument("--evaluate_during_training", type=bool, default=True, help="Evaluate during training")
    parser.add_argument("--evaluate_generated_text", type=bool, default=True, help="Evaluate generated text")
    parser.add_argument("--evaluate_during_training_verbose", type=bool, default=True,
                        help="Verbose evaluation during training")
    parser.add_argument("--use_multiprocessing", type=bool, default=False, help="Use multiprocessing")
    parser.add_argument("--max_length", type=int, default=25, help="Max length")
    parser.add_argument("--manual_seed", type=int, default=4, help="Manual seed")
    parser.add_argument("--save_steps", type=int, default=11898, help="Save steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--output_dir", type=str, default="./exp/template", help="Output directory")
    parser.add_argument("--n_gpu", type=int, default=2, help="Number of GPUs")

    args = parser.parse_args()

    args, _ = parser.parse_known_args()

    train_df = pd.read_csv(args.train_file)
    # train_df = pd.DataFrame(train_data, columns=["input_text", "target_text"])
    print(train_df.head())
    eval_df = pd.read_csv(args.dev_file)

    model_args = {
        "reprocess_input_data": True,
        "overwrite_output_dir": True,
        "max_seq_length": 50,
        "train_batch_size": 4,
        "num_train_epochs": 5,
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

    test_df = pd.read_csv(args.test_file)
    with open("./exp/template/predictions.csv", 'w') as f:
        for text in test_df.input_text:
            f.write(model.predict([text]) + '\n')

    # Use the model for prediction
    # Le public est averti que [pers] Charlotte née Bourgoin [/pers] ,
    # femme - de [pers] Joseph Digiez [/pers] , et [pers] Maurice Bourgoin [/pers] ,
    # enfant mineur représenté par le [pers] sieur Jaques Charles Gicot [/pers] son curateur ,
    # ont été admis par arrêt du [org] Conseil d ' Etat [/org] du [time] 5 décembre 1797 [/time] ,
    # à solliciter une renonciation"

    # Le public est averti que [pers] Charlotte [/pers] née Bourgoin [/pers],
    # femme - de [pers] Joseph Digiez [/pers], et [pers] Maurice Bourgoin [/pers],
    # enfant mineur représenté par le [pers] sieur Jaques Charles Gicot [/pers] son curateur,
    # ont été admis par arrêt du [org] Conseil d'Etat [/org] du 5 décembre 1797 [/org],
    # à solliciter une renonciation

    ENTITY_PATTERN = '\[\w*\](.*?)\[\/\w*\]'

    pred_df = pd.read_csv('data/', header=None)
    pred_df.columns = ['pred_text']

    with open("data/decoded_predictions.tsv", 'w') as f:
        for test_text, pred_text in zip(test_df.input_text, pred_df.pred_text):
            raw_words_i = text.split(' ')
            test_matches = re.findall(ENTITY_PATTERN, test_text, re.DOTALL)
            pred_matches = re.findall(ENTITY_PATTERN, pred_text, re.DOTALL)
            print(pred_matches)

            f.write(model.predict([text]) + '\n')
    # eval_df = pd.DataFrame(eval_data, columns=["input_text", "target_text"])

    # print(model.predict(["Japan began the defence of their Asian Cup title with a lucky 2-1 win against Syria in a Group C championship match on Friday."]))
