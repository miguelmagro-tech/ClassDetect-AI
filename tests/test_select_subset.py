import json
import shutil
import subprocess
import sys
from pathlib import Path


def test_selection_is_reproducible_and_preserves_original_manifest(tmp_path):
    root = tmp_path
    script = root / "scripts" / "select_subset.py"
    script.parent.mkdir()
    shutil.copy2(Path(__file__).resolve().parents[1] / "scripts/select_subset.py", script)
    base = root / "datasets/coco2017"
    (base / "annotations").mkdir(parents=True)
    classes = ["person", "backpack", "book", "cell phone", "laptop",
               "chair", "bottle", "cup", "keyboard", "mouse"]
    data = {
        "categories": [{"id": i + 1, "name": name} for i, name in enumerate(classes)],
        "images": [{"id": i, "file_name": f"{i}.jpg"} for i in range(45)],
        "annotations": [{"image_id": i, "category_id": j + 1}
                        for i in range(45) for j in range(len(classes))],
    }
    (base / "annotations/instances_val2017.json").write_text(json.dumps(data), encoding="utf-8")
    original = base / "selected_images.json"
    original.write_text('{"original": true}', encoding="utf-8")

    for _ in range(2):
        subprocess.run([sys.executable, str(script)], check=True, capture_output=True, text=True)
        output = root / "runs/dataset_preview/selected_images.json"
        manifest = json.loads(output.read_text(encoding="utf-8"))
        assert len(manifest["images"]) == 40
        if _ == 0:
            first = manifest
        else:
            assert manifest == first
        assert original.read_text(encoding="utf-8") == '{"original": true}'
