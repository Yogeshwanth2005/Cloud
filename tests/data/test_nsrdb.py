import textwrap
from aco.data.nsrdb import load_nsrdb_file


def test_load_nsrdb_file_skips_metadata_and_renames(tmp_path):
    p = tmp_path / "nsrdb_golden_2018.csv"
    p.write_text(textwrap.dedent("""\
        Source,Location ID
        NSRDB,479494
        Year,Month,Day,Hour,Minute,GHI,DNI,DHI,Temperature,Wind Speed,Pressure,Relative Humidity
        2018,1,1,0,30,0,0,0,-9.8,0.5,812,66.3
    """))
    df = load_nsrdb_file(str(p))
    assert list(df.columns) == [
        "timestamp", "ghi", "dni", "dhi", "temperature",
        "wind_speed", "pressure", "relative_humidity",
    ]
    assert df.iloc[0]["ghi"] == 0
    assert str(df.iloc[0]["timestamp"]) == "2018-01-01 00:30:00"
