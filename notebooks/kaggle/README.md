# Kaggle launchers

Thin launchers only — install the package, load a YAML config, call one entry point.
No logic lives here (Engineering standards: "No logic that lives only in notebooks").

Shape every notebook follows:

```python
!pip install -q -e /kaggle/working/specheads
from specheads.bench.run_bench import main
main(config_path="configs/medusa_tree.yaml")
```

Notebooks are added from Phase 1, once there is an entry point to call.
