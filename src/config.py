import transformers

MAX_LEN = 512
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 10

BERT_PATH = "../data/bert_base_uncased"
MODEL_PATH = "model.bin"

JIGSAW_COMMENT_TRAIN =   "../data/jigsaw-toxic-comment-train.csv"
JIGSAW_TOX_COM_TRAIN_PROCESS_128 = "../data/jigsaw-toxic-comment-train-processed-seqlen128.csv"

JIGSAW_UNINTENDED_BIAS_TRAIN = "../data/jigsaw-unintended-bias-train.csv"
JIGSAW_UNINTENDED_BIAS_TRAIN_PROCESS_128 = "../data/jigsaw-unintended-bias-processed-seqlen128.csv"

SAMPLE_SUBMISSION = "../data/sample_submission.csv"
TEST = "../data/test.csv"
TEST_PROCESSED_SQLEN128 = "../data/test-processed-seqlen128.csv"

VALIDATION_CSV = "../data/validation.csv"
VALIDATION_PROCESSED_SQLEN128 = "../data/validation-processed-seqlen128.csv"

TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BERT_PATH, do_lower_case=True)
)



