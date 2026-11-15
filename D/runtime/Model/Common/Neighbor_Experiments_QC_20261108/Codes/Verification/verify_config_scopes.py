"""Check factor-config scopes against independent source registries/dictionaries."""
from pathlib import Path
import argparse,json
import pandas as pd
from independent_expectations import expected_scopes


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,required=True)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--config-dir',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    expected=expected_scopes(args.project_root,args.archive)
    config=json.loads((args.config_dir/'experiments.json').read_text(encoding='utf-8'))
    sets=json.loads((args.config_dir/'feature_sets.json').read_text(encoding='utf-8'))['sets']
    rows=[];combos={}
    for ident,spec in config['conditions'].items():
        factor=spec['factors'];combo=(factor['china_east'],factor['japan_east'],factor['west_selection']=='new25')
        assert combo not in combos
        combos[combo]=ident
        wanted=expected['conditions'][combo]
        assert len(spec['station_keys'])==len(set(spec['station_keys']))
        assert set(spec['station_keys'])==wanted['stations'],ident
        assert len(sets[ident])==len(set(sets[ident]))==len(wanted['features'])
        assert set(sets[ident])==wanted['features'],ident
        assert sets[ident][:70]==config['common_features']
        rows.append({'condition':ident,'china_east':combo[0],'japan_east':combo[1],
                     'west25':combo[2],'stations':len(wanted['stations']),'features':len(wanted['features'])})
    assert len(combos)==8 and set(combos)==set(expected['conditions'])
    old_sets=json.loads((args.project_root/'Model/Common/Regional_Experiments_20261107/feature_sets.json').read_text(encoding='utf-8'))['sets']
    assert sets[combos[(False,False,False)]]==old_sets['F']
    registry=pd.read_csv(args.config_dir/'station_registry.csv')
    assert len(registry)==650 and not registry.station_key.duplicated().any()
    assert set(registry.station_key)==set.union(*(value['stations'] for value in expected['conditions'].values()))
    comparisons=json.loads((args.config_dir/'comparisons.json').read_text(encoding='utf-8'))['comparisons']
    factor_positions={'china_east':0,'japan_east':1,'west_selection':2}
    reverse={ident:combo for combo,ident in combos.items()}
    actual_pairs=set();primary=0
    for comparison in comparisons:
        better=comparison['better_condition'];reference=comparison['reference_condition']
        assert better in reverse and reference in reverse
        a=reverse[reference];b=reverse[better]
        changed=[n for n in range(3) if a[n]!=b[n]]
        if len(changed)==1:
            assert not a[changed[0]] and b[changed[0]]
        else:
            assert a==(False,False,False) and b==(True,True,False)
        assert comparison['positive_gain_definition']=='RMSE(reference)-RMSE(better)'
        actual_pairs.add((reference,better))
        primary+=comparison.get('priority')=='primary'
    wanted_pairs=set()
    for combo,ident in combos.items():
        for n in range(3):
            if not combo[n]:
                other=list(combo);other[n]=True
                wanted_pairs.add((ident,combos[tuple(other)]))
    wanted_pairs.add((combos[(False,False,False)],combos[(True,True,False)]))
    assert len(actual_pairs)==len(comparisons)==13 and actual_pairs==wanted_pairs and primary==4
    report={'status':'passed','conditions':rows,'old_F600_feature_list_order_exact':True,
            'station_union':650,'independent_source_station_and_feature_sets':True,
            'west_selection_filter_checked':'188rows -> selectedTrue25, overlapold24=9',
            'comparisons':13,'primary_comparisons':4,'single_factor_edges':12}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
