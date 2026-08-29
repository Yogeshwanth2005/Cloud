from aco.data.fleet import parse_plant_filename


def test_parse_actual_filename():
    meta = parse_plant_filename("Actual_31.85_-110.85_2006_UPV_100MW_5_Min.csv")
    assert meta == {
        "kind": "Actual", "lat": 31.85, "lon": -110.85, "year": 2006,
        "plant_type": "UPV", "capacity_mw": 100.0, "resolution_min": 5,
    }


def test_parse_da_filename():
    meta = parse_plant_filename("DA_31.95_-110.95_2006_DPV_43MW_60_Min.csv")
    assert meta["kind"] == "DA"
    assert meta["plant_type"] == "DPV"
    assert meta["capacity_mw"] == 43.0
    assert meta["resolution_min"] == 60


def test_parse_ha4_filename():
    meta = parse_plant_filename("HA4_31.85_-110.85_2006_UPV_100MW_60_Min.csv")
    assert meta["kind"] == "HA4"
    assert meta["plant_type"] == "UPV"
    assert meta["capacity_mw"] == 100.0
    assert meta["resolution_min"] == 60
