# Trained weights

Place your best checkpoints here (produced by the training pipeline):

- `bigru_best.pth`   ← from your BiGRU run's `best_model/` folder
- `bilstm_best.pth`  ← from your BiLSTM run's `best_model/` folder

Each file is the dict saved by training: `{"model": state_dict, "config": {...}, ...}`.
The app loads `ckpt["model"]` and rebuilds the architecture from `ckpt["config"]`.

If a file is missing, the app still runs in **demo mode** (random weights) so the
interface can be reviewed — predictions are not meaningful until real weights are added.
