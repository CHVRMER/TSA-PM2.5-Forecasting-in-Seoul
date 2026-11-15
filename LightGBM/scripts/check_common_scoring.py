"""시험 데이터를 열지 않고 시간축·Holm·시험 잠금·1회 잠금을 검사한다."""
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from src import common_scoring as c

def main():
    # 결측 없는 경우 기존 Bartlett 식과 수치적으로 같아야 한다.
    a=np.array([1.,2.,3.,1.,4.,2.,1.,3.]);b=np.array([2.,1.,2.,2.,1.,3.,2.,1.])
    d=a*a-b*b;dc=d-d.mean();n=len(d);lag=2
    expected=d.mean()/np.sqrt((dc@dc/n+2*sum((1-k/3)*(dc[k:]@dc[:-k])/n for k in [1,2]))/n)
    assert np.isclose(c.dm_test(a,b,6)[0],expected)
    # 누락 슬롯을 유지한 수동 lag=1 계산.
    a=np.array([1.,np.nan,2.,3.,np.nan,4.]);b=np.array([2.,np.nan,1.,2.,np.nan,1.])
    good=np.isfinite(a);d=a[good]**2-b[good]**2;z=np.zeros(6);z[good]=d-d.mean()
    expected=d.mean()/np.sqrt((z@z+z[1:]@z[:-1])/(len(d)**2))
    assert np.isclose(c.dm_test(a,b,3)[0],expected)
    assert np.allclose(c.holm([.01,.04,.03]),[.03,.06,.06])
    assert np.isclose(c.holm([.01,np.nan])[0],.02)
    with patch.object(c.pq,'read_table',side_effect=AssertionError('시험 데이터 접근 금지')):
        for final in [False,True]:
            try:c.evaluate('test',['M1'],final=final)
            except PermissionError:pass
            else:raise AssertionError('시험 잠금 실패')
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp);pred=root/'pred';pred.mkdir();(pred/'M1.parquet').write_bytes(b'lock-test-only')
        f=root/'frozen.json';lock=root/'once.lock'
        f.write_text(json.dumps({'models':['M1'],'horizons':[6],'no_refit_on_2023':True,'artifacts_sha256':{'pred/M1.parquet':c.sha(pred/'M1.parquet')}}))
        with patch.object(c,'ROOT',root),patch.object(c,'PRED',pred):
            c.authorize_test('test',True,f,['M1'],[6],lock)
            try:c.authorize_test('test',True,f,['M1'],[6],lock)
            except FileExistsError:pass
            else:raise AssertionError('1회 잠금 실패')
    print('PASS: 규칙적/결측 시간축 HAC, Holm, 무승인 시험 접근 차단, 임시 파일의 1회 잠금')
if __name__=='__main__':main()
