# Code and configuration guide

Tai lieu nay giai thich code, tham so va cach chay experiment tren Jetson AGX Orin.

## Entry point

Config va CLI nam trong src/run_experiment.py. Ba model nam trong:

~~~text
src/model_unixcoder.py
src/model_ast_gat.py
src/model_fusion.py
~~~

Lenh mac dinh chay lan luot 4 smell:

~~~bash
python src/run_experiment.py --smell all --model fusion
~~~

Moi smell la mot binary classification task rieng, gom positive va negative cua archive smell do.

## Pipeline

1. Tim dung data/training_data_java/<Smell>.7z.
2. Build/reuse dataset_index.csv.
3. Sample positive/negative theo ty le 1:4, uu tien diversity theo repo.
4. Split train/val/test theo 70/15/15.
5. Extract source mot lan/archive va cache sampled_sources.jsonl.
6. Audit token length, truncate, luu token_cache.pt.
7. Parse Java AST, tao typed graph, luu ast_graphs.pt.
8. Train, validation inference va threshold tuning.
9. Chon checkpoint theo validation MCC, F1, roi epoch som hon.
10. Test bang threshold/checkpoint tot nhat va append results/results.csv.

Neu repo-disjoint split dat dung size 70/15/15 thi code dung repo-aware split. Neu khong, code fallback sang label-stratified split. Cot split_protocol ghi protocol thuc te.

## Model modes

### unixcoder

- Encoder microsoft/unixcoder-base.
- Lay embedding token dau tien.
- Dropout va linear classifier 2 labels.

### ast_gat

- Parse Java bang tree-sitter.
- Node feature: hashed node type, depth, child count, sibling index, named flag, terminal flag.
- Edge: parent/child, sibling, same identifier va control-flow hint.
- GAT layers, global mean pooling va classifier.
- Same-identifier bo qua ten ngan hon 3 ky tu va gioi han 10 occurrence moi ten.

### fusion

- Chay UniXCoder semantic encoder va AST-GAT structural encoder.
- Concatenate hai embedding roi dua qua MLP classifier.
- Day la mode ton compute va memory nhat.

## Main config

Sua cac hang so o dau src/run_experiment.py neu muon thay default:

| Config | Default | Y nghia |
|---|---:|---|
| LANGUAGE | java | Dataset language. C# graph parser chua implement. |
| SMELL | all | Chay ca 4 smells. |
| MODEL_NAME | fusion | unixcoder, ast_gat hoac fusion. |
| SEED | 42 | Sampling, split va PyTorch seed. |
| POSITIVE_SAMPLES | 2000 | Full-mode positive request moi smell. |
| NEGATIVE_SAMPLES | 8000 | Full-mode negative request moi smell. |
| DEV_POSITIVE_SAMPLES | 200 | Dev positive moi smell. |
| DEV_NEGATIVE_SAMPLES | 800 | Dev negative moi smell. |
| TRAIN_RATIO | 0.70 | Train split. |
| VAL_RATIO | 0.15 | Validation split. |
| TEST_RATIO | 0.15 | Test split. |
| MAX_TOKEN_LENGTH | 512 | Token cap sau full-length audit. |
| MAX_AST_NODES | 512 | Node cap moi graph. |
| BATCH_SIZE | 8 | Default batch size. |
| MAX_EPOCHS | 30 | Epoch cap. |
| PATIENCE | 5 | Early stopping patience. |
| LEARNING_RATE | 2e-5 | AdamW learning rate. |
| WEIGHT_DECAY | 0.01 | AdamW weight decay. |
| FOCAL_GAMMA | 2.0 | Focal Loss gamma. |
| WARMUP_RATIO | 0.10 | Warmup fraction cho UniXCoder va Fusion. |
| NUM_WORKERS | 0 | DataLoader workers. |
| DEVICE | auto | CUDA neu available, neu khong thi CPU. |
| GAT_HIDDEN_DIM | 128 | Hidden size moi GAT head. |
| GAT_HEADS | 4 | GAT attention heads. |
| GAT_LAYERS | 2 | So GAT layers. |
| DROPOUT | 0.2 | Model dropout. |

Khong thay doi ba ratio neu report bat buoc 70/15/15.

## CLI parameters

| Argument | Mo ta |
|---|---|
| --language java | Chon language. |
| --smell all | Chay ca 4 smells. |
| --smell ComplexMethod | Debug mot smell. |
| --model fusion | Chon model mode. |
| --model all | Chay lan luot unixcoder, ast_gat va fusion, reuse cung artifact. |
| --member Giang | Gia tri cot member. |
| --hardware "Jetson AGX Orin 64GB" | Ghi hardware vao result CSV. |
| --dev | 200 positive + 800 negative moi smell. |
| --positive N | Override positive request. |
| --negative N | Override negative request; mac dinh positive x4. |
| --batch-size N | Override batch size. |
| --epochs N | Override epoch cap. |
| --prepare-only | Chi tao/reuse artifact, khong train. |
| --eval-only | Load checkpoint va test lai. |
| --force-rebuild-index | Re-list archive. |
| --force-resample | Sample, split va extract lai. |
| --force-rebuild-tokens | Tokenize/audit lai. |
| --force-rebuild-ast | Parse/build graph lai. |
| --no-save-model | Khong ghi checkpoint. |

~~~bash
python src/run_experiment.py --help
~~~

## Recommended Orin runs

Prepare va audit:

~~~bash
python src/run_experiment.py \
  --smell all \
  --model fusion \
  --dev \
  --prepare-only \
  --hardware "Jetson AGX Orin 64GB"
~~~

Dev train:

~~~bash
python src/run_experiment.py \
  --smell all \
  --model fusion \
  --dev \
  --batch-size 4 \
  --epochs 3 \
  --hardware "Jetson AGX Orin 64GB"
~~~

Full train:

~~~bash
python src/run_experiment.py \
  --smell all \
  --model fusion \
  --batch-size 4 \
  --epochs 30 \
  --hardware "Jetson AGX Orin 64GB"
~~~

Theo doi tai nguyen:

~~~bash
tegrastats
watch -n 2 df -h
~~~

Neu memory con du nhieu, thu batch size 8. Neu OOM, giam ve 2. Batch size khong lam thay doi sampling hay split 70/15/15.

## Cache invalidation

- Dataset/archive thay doi: --force-rebuild-index --force-resample.
- Sampling request/seed thay doi: --force-resample.
- MAX_TOKEN_LENGTH hoac tokenizer thay doi: --force-rebuild-tokens.
- AST feature, edge, parser hoac MAX_AST_NODES thay doi: --force-rebuild-ast.
- Model hyperparameter thay doi: train lai checkpoint; data cache co the reuse.

Khong can force rebuild cho moi run. Code tu reuse artifact neu cache hop le.

## Output checks

Sau prepare:

~~~text
artifacts/java/<Smell>/sample_audit.csv
artifacts/java/<Smell>/split_summary.csv
artifacts/java/<Smell>/source_extract_audit.csv
artifacts/java/<Smell>/token_stats.csv
artifacts/java/<Smell>/ast_parse_result.csv
artifacts/java/<Smell>/token_cache_ids.txt
artifacts/java/<Smell>/ast_cache_ids.txt
artifacts/java/<Smell>/prepare_run.log
artifacts/java/<Smell>/<model>_run.log
~~~

Sau train/test:

~~~text
checkpoints/java/<Smell>/<model>_best.pt
results/results.csv
~~~

Khi --smell all, mot model run tao 4 dong result. Khi them --model all, run tao 12 dong: 4 target smells x 3 models. Cot smells luon la target smell cua dong do.
