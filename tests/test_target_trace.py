"""Separate numerical drift/token identity from incorrect greedy verification."""
import struct
import array
import pytest
from tools.bench.analyze_qwen_target_trace import audit_verification,compare,tensor,NonfiniteLogitsError


def bf16(values):
    return [struct.unpack('<I',struct.pack('<f',v))[0]>>16 for v in values]


def test_near_tie_changes_token_without_verification_error():
    plain=bf16([25.125,25.125,0])
    verify=bf16([25.125,25.375,0])
    drift=compare(plain,verify)
    assert drift['plain_top5'][0][0]==0
    assert drift['verify_top5'][0][0]==1
    assert drift['plain_margin']==0
    result=audit_verification([100],verify,3,[1],[],[1],1,0,1)
    assert result['errors']==[]  # a token difference is not a licensing error


@pytest.mark.parametrize('width',[2,4,6,16])
def test_full_prefix_bonus_and_masked_tail(width):
    logits=bf16([0,2,1]*width)
    good=audit_verification(list(range(50,50+width)),logits,3,[1]*width,
                            [1]*(width-1),[1]*width,width,width-1,width)
    assert good['errors']==[]
    # Only two columns remain at the output limit; inactive targets/drafts are irrelevant.
    tail=audit_verification([50,51]+[51]*(width-2),logits,3,[1]*width,
                            [1]*(width-1),[1]*width,2,1,2)
    assert tail['errors']==[]


@pytest.mark.parametrize('change,expected',[
    ({'targets':[0,1]},'wrong_target_argmax:0'),
    ({'licensed':[0,1]},'wrong_licensed_token:0'),
    ({'accepted':0,'count':1},'wrong_accepted_prefix'),
    ({'count':1},'wrong_bonus_count'),
    ({'count':3},'licensed_count_out_of_bounds')])
def test_actual_verification_errors_are_reported(change,expected):
    args=dict(positions=[50,51],logits=bf16([0,2,1]*2),vocab=3,
              targets=[1,1],drafts=[1],licensed=[1,1],count=2,accepted=1,valid_width=2)
    args.update(change)
    assert expected in audit_verification(**args)['errors']


def test_rejected_column_target_argmax_is_audited_too():
    r=audit_verification([50,51],bf16([0,2,1]*2),3,[1,0],[2],[1,0],1,0,2)
    assert r['errors']==['wrong_target_argmax:1']


def test_nonfinite_logits_are_numerical_failures_not_token_identity_failures():
    with pytest.raises(NonfiniteLogitsError,match='nonfinite'):
        audit_verification([1],bf16([float('nan'),0]),2,[0],[],[0],1,0,1)


def test_fp32_head_trace_preserves_vocabulary_and_argmax(tmp_path):
    values = array.array('f', [1.0078125, 1.015625, -3.25, 2.0, 0.0, 1.0])
    (tmp_path/'logits.bin').write_bytes(values.tobytes())
    decoded = tensor(tmp_path, {'name':'logits', 'dtype':1, 'file':'logits.bin'})
    assert list(decoded) == list(values)
    assert len(decoded) == 6
    assert audit_verification([50,51], decoded, 3, [1,0], [1], [1,0], 2,1,2)['errors'] == []
    drift = compare(decoded[:3], array.array('f', [1.03125,1.015625,-3.25]))
    assert drift['plain_top5'][0][0] == 1
    assert drift['verify_top5'][0][0] == 0


def test_explicit_bf16_trace_and_unknown_dtype(tmp_path):
    (tmp_path/'logits.bin').write_bytes(array.array('H',bf16([2,1])).tobytes())
    assert list(tensor(tmp_path, {'name':'logits','dtype':0,'file':'logits.bin'})) == bf16([2,1])
    with pytest.raises(ValueError,match='unsupported trace dtype'):
        tensor(tmp_path, {'name':'logits','dtype':6,'file':'logits.bin'})
