# DeepLearningSmells

Code nay dat truc tiep trong project `DeepLearningSmells`, dung dataset co san o `data/`.

Tai lieu chi tiet ve code, CLI va cac bien config: `CODE_CONFIG_GUIDE.md`.

## Cau truc chinh

~~~text
src/
  model_unixcoder.py
  model_ast_gat.py
  model_fusion.py
  run_experiment.py
artifacts/
  java/<Smell>/
    dataset_index.csv
    sampled_dataset.csv
    sample_audit.csv
    splits.csv
    split_summary.csv
    source_extract_audit.csv
    sampled_sources.jsonl
    source_cache_ids.txt
    token_cache.pt
    token_cache_ids.txt
    token_stats.csv
    ast_parse_result.csv
    ast_graphs.pt
    ast_cache_ids.txt
    prepare_run.log
    <model>_run.log
results/results.csv
checkpoints/java/<Smell>/<model>_best.pt
~~~

## Jetson AGX Orin

May muc tieu:

- Jetson AGX Orin Developer Kit, `aarch64`
- Jetson Linux R36.5.0, CUDA 12.6
- 64 GB unified RAM, 12 CPU cores
- Khoang 29 GB disk trong

Phan cung du de train UniXCoder, AST-GAT va Fusion cua project. Nen chay `--dev` truoc, theo doi `tegrastats`, va bat dau Fusion voi `--batch-size 4` neu muon de du memory headroom. Full run 4 smells se cham hon GPU desktop/datacenter; 29 GB disk du nhung can theo doi cache model, artifact va checkpoint.

## Tao venv

Khuyen nghi Python 3.10 tren Ubuntu/Jetson:

~~~bash
sudo apt update
sudo apt install -y python3.10-venv p7zip-full

cd ~/DeepLearningSmells
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
~~~

## Chay

~~~bash
source .venv/bin/activate
python src/run_experiment.py --language java --smell all --model fusion --member Giang
~~~

Dev mode chay nho truoc:

~~~bash
python src/run_experiment.py --smell all --model fusion --dev
~~~

Chay ca 3 model modes trong mot command:

~~~bash
python src/run_experiment.py --smell all --model all --dev
~~~

Artifact data/token/AST duoc prepare mot lan moi smell, sau do reuse cho unixcoder, ast_gat va fusion.

`--dev` request 200 positive + 800 negative cho moi smell. Full mode request 2000 positive + 8000 negative. Sau khi doc index, code clamp theo so mau thuc co va ghi ro vao `sample_audit.csv`.

`--smell all` la mac dinh va se chay lan luot 4 binary tasks:

- `ComplexMethod`
- `ComplexConditional`
- `FeatureEnvy`
- `MultifacetedAbstraction`

Chi prepare artifact cho ca 4 smells, chua train:

~~~bash
python src/run_experiment.py --smell all --prepare-only
~~~

Eval/test lai bang checkpoint da luu:

~~~bash
python src/run_experiment.py --smell all --model fusion --eval-only
~~~

## Log va audit

Moi smell/model co log rieng:

~~~text
artifacts/java/<Smell>/<model>_run.log
~~~

Log ghi cac moc chinh: start run, build/reuse index, sample, split, extract source, tokenize, parse AST, train epoch, save checkpoint, test result, va traceback neu loi.

CSV audit:

- `sample_audit.csv`: requested/available/selected positive/negative, tong sample, so repo.
- `split_summary.csv`: size train/val/test, label count tung split, so repo tung split.
- `source_extract_audit.csv`: trang thai extract cua tung source; file thieu/khong doc duoc se bi loai dong bo va split lai.
- `source_cache_ids.txt`: sidecar ID nho de kiem tra source cache ma khong scan lai JSONL.
- `token_stats.csv`: token length tung sample, sort giam dan; dung de xem sample dai nhat va so sample bi truncate.
- `ast_parse_result.csv`: trang thai parse AST, so node, so edge.
- `results/results.csv`: bang ket qua cuoi cung theo schema project.

## Sampling va split

Sampling lay positive va negative rieng theo ty le 1:4. Thay vi `random.sample()` toan cuc, code sampling vong tron theo `repo` de tranh viec 2-3 repo lon chiem het negative.

`repo` duoc suy tu ten file trong archive. Vi du:

~~~text
ComplexConditional/Positive/247687009_soa_io.goudai..._QuartzServiceImpl0start.code
~~~

Repo id se la `247687009`.

Split size luon theo 70/15/15. Code uu tien repo-aware de giam leakage, nhung chi chap nhan neu split repo-disjoint cung dat dung size 70/15/15. Neu khong dat, code fallback ve split stratified by label. Protocol thuc te duoc luu trong `splits.csv` va ghi dong vao `results/results.csv`.

## Cache source

Index archive uu tien dung:

~~~bash
7z l -slt <archive>
~~~

Code chi doc 4 archive `.7z` hien thi la WinRAR archive trong Explorer:

~~~text
ComplexMethod.7z
ComplexConditional.7z
FeatureEnvy.7z
MultifacetedAbstraction.7z
~~~

File extensionless `*_1`, `*_2` va `readme.txt` bi bo qua, khong duoc list, ghep hay extract.

Source cache uu tien extract bang 7z CLI mot lan cho moi archive:

~~~bash
7z x <archive> -o<tmp_dir> -scsUTF-8 @targets.txt
~~~

Neu may khong co `7z`, `7za`, hoac `7zr`, code fallback sang `py7zr` one-shot extraction cho archive do va ghi ro vao log.

File source ho tro duoi:

~~~text
.code, .java, .cs, .txt
~~~

## Token audit

`MAX_TOKEN_LENGTH = 512` la gioi han input experiment. Code van tokenize full truoc de tinh `token_length`, sau do moi truncate/cache. Sau prepare, xem:

~~~text
artifacts/java/<Smell>/token_stats.csv
~~~

File nay cho biet sample dai nhat, `cached_length`, va `is_truncated`.

Rieng FeatureEnvy, token input duoc prepend ten containing class parse tu filename. Source cache van giu code nguyen ban; `token_stats.csv` co `class_name` va `class_context_added` de audit. Token cache metadata tu dong rebuild cache khi schema/context thay doi.

## AST graph

Node feature gom:

- hash bucket cua `node.type`
- `depth`
- `child_count`
- `sibling_index`
- `is_named`
- `is_terminal`

Edge type gom:

- parent -> child
- child -> parent
- next sibling
- previous sibling
- same identifier
- control-flow hint
- member receiver cho method invocation va field access

`same_identifier` bo qua ten ngan hon 3 ky tu va chi dung toi da 10 occurrence moi ten de gioi han edge/memory. Day chua phai CFG/DFG compiler-grade, nhung manh hon AST parent-child thuan va phu hop baseline.

## Threshold va loss

Moi epoch tune threshold tren validation probabilities theo MCC truoc, F1 sau. Neu tie MCC/F1, threshold gan 0.5 hon duoc chon. Checkpoint luu threshold tot nhat.

Focal Loss tinh CE khong weight de lay `pt`, sau do moi nhan `alpha_t`, dung hon cong thuc focal loss chuan.

UniXCoder va Fusion dung linear learning-rate schedule voi warmup 10% optimizer steps. AST-GAT giu learning rate phang.

## CSV result

`results/results.csv` xuat schema:

~~~csv
member,task_name,dataset_name,language,smells,input_type,split_protocol,seed,train_size,validation_size,test_size,model,checkpoint_selection,hardware,runtime,accuracy,precision,recall,f1,mcc,roc_auc,notes
~~~

Mac dinh `member=Giang`. Khi chay `--smell all`, mot model append 4 dong, moi dong la metric cua mot target smell. Cot `smells` ghi target smell cua chinh dong do. `--smell all --model all` append 12 dong: 4 smells x 3 models.

Neu chay tren Kaggle:

~~~bash
python src/run_experiment.py --smell all --model fusion --hardware "Kaggle T4 x2"
~~~

## Luu model

Mac dinh co luu best checkpoint de eval/test lai. UniXCoder/Fusion checkpoint thuong vai tram MB; AST-GAT nho hon. Khong muon luu:

~~~bash
python src/run_experiment.py --smell all --model fusion --no-save-model
~~~

Khi do van co `results/results.csv`, nhung muon eval lai thi phai train lai.


