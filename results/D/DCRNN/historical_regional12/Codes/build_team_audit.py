from pathlib import Path
import json
import pandas as pd

OUT = Path(__file__).resolve().parent
DOC = json.loads((OUT / 'team06_docx_extracted.json').read_text(encoding='utf-8'))
PASTED = Path(r'C:/Users/kimjy/.codex/attachments/a5ce9bac-7d59-4905-aefa-21fb9ef1a55c/붙여넣은 텍스트.txt')
LOCAL = Path(r'C:/Users/kimjy/Desktop/Time Series Analysis/Project/Revised_Project/Model/LightGBM/Neighbor_Expansion_QC_20261108/Results/Validation_Reference_20261108')

report = {
    'horizons': [3, 6, 12, 24],
    'A_source': {
        'path': r'C:/Users/kimjy/Downloads/팀공유_계획A_결과.pdf',
        'split': 'validation2023 only', 'metric_tables_pages': [1, 2, 3],
        'holm_family': 112, 'hac_lag': 'max(h/3-1,0)',
        'scope': 'KR67 or W24 plus near/far CN among CN61; A-used v5 files claimed identical to v3',
        'raw_predictions_available': False,
    },
    'A_validation_RMSE': {
        'Persistence': [7.18, 9.97, 13.11, 15.20],
        'SARIMA': [7.15, 9.73, 12.05, 13.52],
        'SARIMAX_WNF': [6.88, 9.42, 11.61, 12.94],
    },
    'ABC_docx_source': {
        'path': r'C:/Users/kimjy/Downloads/Team06_진행보고_20261109.docx',
        'split': 'validation2023 only', 'metric_table_index': 1,
        'scope': 'D67; DN=KR514; F=600 external; ADD8; DROP15; CTRL3',
        'seed_aggregation': '11/22/33 ensemble; GCN DROP initially seed11 only',
        'raw_repo_reference': 'github.com/dain77-d/Time-Series-Data-Analysis-and-Practice-Project (results/A,B,C)',
        'raw_predictions_in_attachment': False,
    },
    'ABC_docx_validation_table': DOC['tables'][1]['rows'],
    'ABC_docx_additional_tables': DOC['tables'][2:],
    'B_pasted_source': {
        'path': str(PASTED), 'test_section_lines': [1535, 1585],
        'validation_rows': [2509, 2504, 2497, 2486],
        'test_rows': [2840, 2835, 2828, 2816],
        'seed_count': 3, 'test_holm_family': 144, 'hac_lag': 'h/3',
        'scope': 'M1 KR67; M2 adds CN21; M3 CN61; M4 JP24+MN1 (153 external maximum)',
        'M3T_scope_caveat': 'M3T deliberately retains all-country trajectory aggregate including JP/MN (lines 665,1131+).',
        'version_caveat': 'This experiment differs from the later regional DN/F600 report and local F634 reference.',
    },
    'B_pasted_validation_RMSE': {
        'M1': [5.9726,8.3359,10.6750,12.8561], 'M2': [5.9656,8.2037,10.2866,12.0750],
        'M3': [5.9590,8.1243,10.2968,11.9586], 'M4': [6.0637,8.1915,10.4746,12.0205],
        'M3T': [5.8842,7.8771,10.1486,11.9710], 'M4T': [5.9273,7.9792,10.1869,12.0709],
        'Adaptive_wind': [5.9235,8.1047,10.2331,11.8946],
        'Adaptive_route': [5.9312,8.1082,10.2158,11.8989],
    },
    'B_pasted_test_RMSE': {
        'Persistence': [6.8127,9.1299,12.1242,14.0856],
        'M1': [5.3192,7.2113,9.5125,11.2354], 'M2': [5.3627,7.2047,9.3015,11.0256],
        'M3': [5.4093,7.3558,9.4744,11.0122], 'M4': [5.3772,7.3620,9.5096,10.9795],
        'M3T': [5.3100,7.1228,9.2843,10.8219], 'M4T': [5.3435,7.1115,9.2748,10.9175],
        'Adaptive_wind': [5.3490,7.3243,9.4378,10.9734],
        'Adaptive_route': [5.3519,7.3489,9.3901,10.9848],
    },
    'B_test_reported_significance': {
        'M3T_vs_M3_h6': {'delta_RMSE': -0.2330, 'ci': [-0.3486,-0.1437], 'holm_p': 0.000376},
        'M4T_vs_M4_h6': {'delta_RMSE': -0.2504, 'ci': [-0.3584,-0.1505], 'holm_p': 0.0000371},
        'any_vs_M1': 'None passed both CI and Holm across four horizons',
        'all_vs_persistence': 'All eight models passed all four horizons',
    },
    'comparability': {
        'A': '2023 rounded descriptive comparison; no raw row keys/predictions or 2024 result.',
        'DOCX_BC': '2023 descriptive only: F600 vs DCRNN F634; three-seed ensemble vs seed11; representations differ. No DOCX2024.',
        'Pasted_B': 'Both years exist, but eligible rows differ, KR67 vs KR514, foreign scope and ensembles differ. No cross-model significance claim without raw fixed-row rescoring.',
        'Local_LGBM_reference': '2023 raw score_rows available; F634 has matching external universe. Check exact keys/y then paired rescore if needed. This is a local reference, not teammate B own result.',
    },
}

metrics = pd.read_csv(LOCAL / 'Common_Score/metrics.csv')
selected = metrics[(metrics.input_condition == 'F634_BOTH_W24') &
                   ((metrics.prediction_kind == 'ensemble') |
                    ((metrics.prediction_kind == 'seed') & (metrics.seed == 11)))]
report['local_LGBM_F634_reference'] = {
    'path': str(LOCAL),
    'grid_sha256': '6a7dac480d621745a8211b85ba4138a90953ad2a2ff5d417ee0a1aaf348e13db',
    'metrics': json.loads(selected[['horizon','prediction_kind','seed','n','RMSE','MAE','MBE','R2']].to_json(orient='records')),
    'score_rows': str(LOCAL / 'Common_Score/score_rows.parquet'),
}
(OUT / 'team_results_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print('Saved team_results_audit.json')
