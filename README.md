# TSA-PM2.5-Forecasting-in-Seoul
Time Series Analysis PM2.5 Forecasting in Seoul Project

## 모델 C (GRU / GCN-GRU)

권역별 관측이 서울 중구 PM2.5 예측(3·6·12·24시간)에 주는 가치를 그래프 순환신경망으로 검증했습니다. 코드·결과 설명은 [C_graph/README.md](C_graph/README.md), 결과 전체는 [results/C/RESULTS_C.md](results/C/RESULTS_C.md)에 있습니다.

| 모델 (2023 검증 RMSE) | 3 h | 6 h | 12 h | 24 h |
|---|---|---|---|---|
| Persistence | 7.18 | 9.97 | 13.11 | 15.20 |
| C_GRU | 6.19 | 8.28 | 10.43 | 11.82 |
| C_GCNGRU | 6.76 | 8.98 | 10.99 | 12.28 |
