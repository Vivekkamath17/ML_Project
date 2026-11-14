import polars as pl

ass = pl.read_csv("data/raw/assessments.csv").filter(pl.col("code_module") == "FFF")
sa = pl.read_csv("data/raw/studentAssessment.csv")
sa_ids = set(sa["id_assessment"].unique().to_list())

print("FFF assessments:")
for r in ass.iter_rows(named=True):
    in_sa = r["id_assessment"] in sa_ids
    print(f"  {r['code_presentation']} {r['id_assessment']} {r['assessment_type']} date={r['date']} weight={r['weight']} in_sa={in_sa}")
