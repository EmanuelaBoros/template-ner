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
            "--do_train",
            action="store_const",
            const=True
        )
        parser.add_argument(
            "--do_test",
            action="store_const",
            const=True
        )
        parser.add_argument(
            "--do_eval",
            action="store_const",
            const=True
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

        if args.do_train:
            train_df = pd.read_csv(args.train_file)
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
                args={**args},
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
        ENTITY_PATTERN = '(\[\w*\])(.*?)(\[\/\w*\])'

        def preprocess(text):
            text = re.sub("([.,'!?()])", r' \1 ', text)
            return re.sub(' +', ' ', text)


        def subfinder(mylist, pattern):
            return list(filter(lambda x: x in pattern, mylist))


        def find_sub_list(sl, l):
            results = []
            sll = len(sl)
            for ind in (i for i, e in enumerate(l) if e == sl[0]):
                if l[ind:ind + sll] == sl:
                    results.append((ind, ind + sll - 1))

            return results

        if args.do_test:
            test_df = pd.read_csv('data/HIPE-2022-v2.1-hipe2020-test-fr_bart.csv')
            with open('data/predictions_10.csv', 'r') as f:
                pred_texts = f.readlines()

            with open("data/decoded_predictions.tsv", 'w') as f:
                for input_text, test_text, pred_text in zip(test_df.input_text, test_df.target_text, pred_texts):

                    tokens = test_text.split()
                    tokens_with_predictions = test_text.split()

                    test_matches = re.findall(ENTITY_PATTERN, test_text, re.DOTALL)
                    pred_matches = re.findall(ENTITY_PATTERN, pred_text, re.DOTALL)

                    test_en_types = [element[0] for element in test_matches]
                    pred_en_types = [element[0] for element in pred_matches]

                    test_ens = [preprocess(element[1]) for element in test_matches]
                    pred_ens = [preprocess(element[1]) for element in pred_matches]

                    for pred_en in pred_ens:
                        found_ens = find_sub_list(pred_en.split(), tokens)
                        # for found_en in founds_ens.
                        if len(found_ens) == 0:
                            print('problem ', pred_en, '-->', test_ens)

                    # print('--'*10)
                    # print(test_ens)
                    # not_in_test = [element for element in pred_ens if element not in test_ens]
                    # not_in_pred = [element for element in test_ens if element not in pred_ens]
                    #
                    # print(not_in_test, '--->', not_in_pred)

                    # import pdb;pdb.set_trace()

                    # f.write(model.predict([text]) + '\n')
            # eval_df = pd.DataFrame(eval_data, columns=["input_text", "target_text"])

            # print(model.predict(["Japan began the defence of their Asian Cup title with a lucky 2-1 win against Syria in a Group C championship match on Friday."]))
