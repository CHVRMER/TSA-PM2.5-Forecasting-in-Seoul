"""Pure-metadata restore checks; no PyTorch state or model is loaded.

The full restore verifier additionally tests checkpoint corruption and recovery
on a machine with PyTorch. This check executes the real contract validator's AST.
"""
from pathlib import Path
import ast
import copy
import hashlib
import json
import re
import tempfile

HERE=Path(__file__).resolve().parent

def main():
    tree=ast.parse((HERE/'restore_verified_backup.py').read_text(encoding='utf-8'))
    wanted={'RestoreError','_digest','_required_json','_model_contract'}
    selected=[node for node in tree.body if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and node.name in wanted]
    namespace={'json':json,'hashlib':hashlib,'re':re,'TRAINING_VERSION':'seoul-gwn-m1-m4t-h48-training-20261112-v1',
               'MODEL':'GraphWaveNet_distance_fixed_plus_adaptive_H48'}
    exec(compile(ast.Module(body=selected,type_ignores=[]),'real_restore_contract_validator','exec'),namespace)
    digest=namespace['_digest'];validate=namespace['_model_contract'];error=namespace['RestoreError']
    with tempfile.TemporaryDirectory(prefix='.restore_schema_',dir=HERE) as temporary:
        base=Path(temporary);run='Conditions/M1/run1';seed=run+'/Models/GraphWaveNet_distance_fixed_plus_adaptive_H48/seed11'
        p={'node_order':['KR:111121']+[f'KR:S{i}' for i in range(634)],
           'context_names':[f'context{i}' for i in range(375)],
           'pm_mean':[0.]*635,'pm_std':[1.]*635,'pm_valid_count':[2]*635,
           'context_mean':[0.]*375,'context_std':[1.]*375,'context_valid_count':[2]*375}
        a={'implementation':'gwn-seoul-h48-rf61-v1','input_dim':2,'context_dim':750,
           'target_index':0,'horizons':[3,6,12,24,36,48],'output_steps':48,'receptive_field_hours':61,
           'architecture_profile':'project48','blocks':4,'layers_per_block':4,'kernel_size':2,
           'residual_channels':32,'dilation_channels':32,'skip_channels':256,'end_channels':512,
           'adaptive_embedding_dim':10,'fixed_support_count':2,'graph_mode':'fixed_plus_adaptive',
           'graph_dropout':0.3,'context_dropout':0.2,'padding_policy':'scaled_pm_zero_missing_one'}
        c={'version':namespace['TRAINING_VERSION'],'model':namespace['MODEL'],'data':'1'*64,'graph':'2'*64,
           'preprocessing':digest(p,compact=False),'architecture':a,
           'training':{'input_condition':'M1','seed':11,'mode':'full','experiment_bundle':'gwn-vast-m1-m4t-h48-v1'}}
        r={'version':c['version'],'data':c['data'],'preprocessing':c['preprocessing'],'training':dict(c['training'])}
        files={seed+'/contract.json':c,seed+'/preprocessing.json':p,run+'/experiment_config.json':r}
        def save(records):
            for relative,value in records.items():
                path=base/relative;path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
        meta={'seed':seed,'run':run,'condition':'M1','seed_number':11}
        save(files);assert validate(base,files,meta)==c
        mutations=[('wrong_model',lambda x:x[seed+'/contract.json'].update(model='GraphWaveNet_distance_fixed_plus_adaptive')),
                   ('old_bundle',lambda x:x[seed+'/contract.json']['training'].pop('experiment_bundle')),
                   ('wrong_layers',lambda x:x[seed+'/contract.json']['architecture'].update(layers_per_block=2)),
                   ('wrong_context',lambda x:x[seed+'/contract.json']['architecture'].update(context_dim=106)),
                   ('wrong_horizons',lambda x:x[seed+'/contract.json']['architecture'].update(horizons=[3,6])),
                   ('old_four_horizons',lambda x:x[seed+'/contract.json']['architecture'].update(horizons=[3,6,12,24])),
                   ('old_decoder_steps',lambda x:x[seed+'/contract.json']['architecture'].update(output_steps=24)),
                   ('old_training_version',lambda x:x[seed+'/contract.json'].update(version='seoul-dcrnn-m1-m4t-training-20261111-v1')),
                   ('dcrnn_h48_model',lambda x:x[seed+'/contract.json'].update(model='DCRNN_trajectory_directed_H48')),
                   ('wrong_RF',lambda x:x[seed+'/contract.json']['architecture'].update(receptive_field_hours=13)),
                   ('wrong_dropout',lambda x:x[seed+'/contract.json']['architecture'].update(context_dropout=0.)),
                   ('wrong_padding',lambda x:x[seed+'/contract.json']['architecture'].update(padding_policy='zeros')),
                   ('wrong_seed',lambda x:x[seed+'/contract.json']['training'].update(seed=22)),
                   ('wrong_run_bundle',lambda x:x[run+'/experiment_config.json']['training'].pop('experiment_bundle'))]
        for name,mutate in mutations:
            values=copy.deepcopy(files);mutate(values);save(values)
            try:validate(base,values,meta)
            except error:pass
            else:raise AssertionError('Contract guard missed '+name)
    print(json.dumps({'restore_schema_tests':'passed','real_metadata_validator_executed':True,
                      'valid_contract_accepted':True,'rejected_mutations':len(mutations),
                      'pytorch_checkpoint_load_tested':False}))

if __name__=='__main__':main()
