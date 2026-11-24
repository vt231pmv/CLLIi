# evidence/: докази реального використання

| Файл | Що це | Як відтворити |
|---|---|---|
| `real_runs/lab6_before.txt`, `lab6_after.txt` | `labcheck` на ноутбуках лаби 6: як вони були після запуску в Colab (до доопрацювання) і фінальна версія; у заголовку SHA-256 вхідних файлів | `python solution/scripts/real_runs.py --evidence evidence` (потрібні ті самі ноутбуки) |
| `real_runs/lab5_before.txt`, `lab5_after.txt` | те саме для лаби 5 | там само |
| `real_runs/mkr1_self_check.txt` | перевірка самої папки `mkr1` перед здачею | `cd solution && python -m labcheck check .. --spec specs/mkr1.json` |
| `history.jsonl` | історія запусків: час, мітка, помилки, попередження, час виконання, токени | дописується автоматично з `--history` |
| `timing.json` | час 10 повторних запусків на кожному вході | `scripts/real_runs.py` |
| `bench_static.txt`, `bench_static.json` | бенчмарк підкинутих дефектів (85 мутантів) | `python -m labcheck bench static` |
| `charts/before_after.png` | графік «до / після» | `python solution/make_chart.py` |
| `ai_run/*` | AI-прогін: журнал, `bench_ai.json`, перевірка `mkr1` з AI-суддею | `python solution/demo_run.py` |
| `dev_issues.md` | реальні проблеми, знайдені під час розробки | — |
