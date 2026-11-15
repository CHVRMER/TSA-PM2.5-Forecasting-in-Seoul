"""기존 명시적 Feature 목록과 역할 검사를 재사용한다."""
import csv
import json


def feature_names(folder, set_name, expected_count=None):
    sets = json.loads((folder / "feature_sets.json").read_text(encoding="utf-8"))["sets"]
    if set_name in sets:
        names = sets[set_name]
    else:
        definition_path = folder.parents[1] / "configs/feature_groups.json"
        definition = json.loads(definition_path.read_text(encoding="utf-8"))[set_name]
        if "union_feature_sets" in definition:
            names = list(dict.fromkeys(c for key in definition["union_feature_sets"] for c in sets[key]))
        else:
            base = sets[definition["base_feature_set"]]
            candidates = sets[definition["additional_feature_set"]]
            keys = set(definition["station_keys"])
            with (folder / "feature_dictionary.csv").open(encoding="utf-8-sig", newline="") as f:
                dictionary = {r["name"]: r for r in csv.DictReader(f)}
            additional = [c for c in candidates if json.loads(dictionary[c]["parameters_json"]).get("station_key") in keys]
            if len(additional) != 16 * len(keys):
                raise ValueError("관측소별 16개 이력 Feature 구성이 다릅니다.")
            names = list(dict.fromkeys(base + additional))
        excluded = definition.get("excluded_features", [])
        if len(excluded) != len(set(excluded)) or not set(excluded) <= set(names):
            raise ValueError("제외 목록의 중복 또는 원본 목록 밖 Feature")
        names = [c for c in names if c not in set(excluded)]
    with (folder / "feature_dictionary.csv").open(encoding="utf-8-sig", newline="") as f:
        roles = {r["name"]: r["role"] for r in csv.DictReader(f)}
    if not names or len(names) != len(set(names)) or any(roles.get(n) != "feature" for n in names):
        raise ValueError("명시적 입력 목록에 중복 또는 비입력 열이 있습니다.")
    if expected_count is not None and len(names) != expected_count:
        raise ValueError("승인된 Feature 수와 다릅니다.")
    return list(names)
