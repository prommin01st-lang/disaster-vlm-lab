set -e
nvidia-smi --query-gpu=name,memory.total --format=csv
python -c "import transformers, peft, trl, torch; print(transformers.__version__, peft.__version__, trl.__version__, torch.cuda.is_available())"
python -c "from dvl.prompt import SYSTEM_PROMPT; print(len(SYSTEM_PROMPT))"
echo ok > out/smoke.txt
