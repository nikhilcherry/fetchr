from fetchr import schema


def test_schema_constants_match_arvyo_pipeline_contract():
    # These must match arvyo-pipeline/arvyo/contract.py exactly -- this
    # test is the tripwire for schema.py's manual-sync obligation.
    assert schema.SCHEMA_VERSION == "1.0"
    assert schema.REQUIRED_ARRAYS == ["time", "flux", "flux_err"]
    assert schema.OPTIONAL_ARRAYS == ["flux_raw"]
    assert schema.REQUIRED_META == ["tic_id", "label", "sector"]
    assert schema.OPTIONAL_META == [
        "period_days", "epoch_btjd", "crowdsap", "mission",
        "augmented", "injection_params",
    ]
    assert schema.LABELS == ["planet", "eb", "blend", "starspot", "null", "unknown"]
