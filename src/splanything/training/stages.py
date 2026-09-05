"""Training stage name constants for callbacks."""

TRAIN_START = "train_start"
TRAIN_END = "train_end"
EPOCH_START = "epoch_start"
EPOCH_END = "epoch_end"
PRE_STEP = "pre_step"
BATCH_START = "batch_start"
BATCH_END = "batch_end"
STAGES = [
    TRAIN_START,
    TRAIN_END,
    EPOCH_START,
    EPOCH_END,
    BATCH_START,
    BATCH_END,
    PRE_STEP,
]
