# Graph WaveNet H48 private common scorer

3/6/12/24/36/48h를기존표준출력6종의동일schema로채점합니다.
원래Model/Common과DCRNN_H48전용채점기는수정하지않습니다.
ensure_h48_grid는Model캐시에정확한6지평평가격자를생성하고기존4지평metadata/정답/기준선을그대로보존합니다.
36/48h의seasonal baseline은t−12/t로미래참조를막습니다.
전체6조건seed11은84jointtests입니다. seed별성능과raw-unit앙상블은분리합니다.
