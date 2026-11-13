#!/usr/bin/env bash
# Установка зависимостей Heretic одной командой.
# Ставит базовое (S0/S1) + DPO-стек (S2) и чинит torchaudio после unsloth.
set -euo pipefail

pip install -r requirements.txt
pip install -r requirements-dpo.txt

# Unsloth обновляет torch и ломает torchaudio (нет совместимого wheel).
# Текстовая модель в torchaudio не нуждается — удаляем.
pip uninstall -y torchaudio || true

echo "Готово."
echo "Для S2 (LoRA DPO) не забудьте войти в Hugging Face (gated датасет toxic-dpo):"
echo "  huggingface-cli login"
