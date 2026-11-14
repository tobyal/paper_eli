"""Preserve and plot the user's 20-row table; no inference of new evaluation data."""
from pathlib import Path
from decimal import Decimal
import copy,csv,hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'industrial_swap_v2/outputs/paper_figures_v4/F08_localization_revised'
OUT=BASE/'user_results_20'
def read(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,indent=2),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    source=Path(__file__).with_name('f08_user_results_20.json');raw=read(source);rows=raw['rows']
    assert [r[0] for r in rows]==[f'{i:02d}' for i in range(1,21)]
    values=[[Decimal(r[k]) for r in rows] for k in [1,2]]
    assert all(0<=v<=1 for vv in values for v in vv)
    means=[sum(v)/Decimal(len(v)) for v in values]
    assert [str(v.quantize(Decimal('.0001'))) for v in means]==raw['supplied_mean']
    OUT.mkdir(exist_ok=True);shutil.copy2(source,OUT/'supplied_results.json')
    metrics=copy.deepcopy(read(BASE/'metrics.json'))
    metrics['per_case']=[dict(label='M1-'+r[0],case_label=r[0],text_only_reported_iou=float(r[1]),
        photo_assisted_reported_iou=float(r[2]),source='user_supplied_result_table',
        note='User-adjusted value' if r[0]=='06' else '') for r in rows]
    metrics['summary'].update(evaluation_mode='user_supplied_results',reported_case_count=20,
        reported_iou_source='supplied_results.json',new_per_view_mask_evaluation=False,
        panel_d_cohort='unchanged earlier ten confirmed M1 cases',
        case_id_mapping='The twenty supplied labels are not asserted to be the historical archived-case numbering.')
    for method,mean in zip(['text_only','photo_assisted'],means):
        metrics['summary'][method]['reported_mean_iou']=float(mean)
        metrics['summary'][method]['reported_mean_iou_exact']=str(mean)
    save(OUT/'metrics.json',metrics);save(OUT/'summary.json',metrics['summary'])
    with (OUT/'iou_results_20cases.csv').open('w',newline='',encoding='utf8') as f:
        writer=csv.writer(f);writer.writerow(raw['columns']);writer.writerows(rows)
        writer.writerow(['Mean',*raw['supplied_mean']])
    for name in ['case_index.json','instance_audit.json']:shutil.copy2(BASE/name,OUT/name)
    sources=read(BASE/'sources.json');sources.update(user_supplied_iou_table=dict(path=str(source),sha256=sha(source)),
        immutable_original_figures={n:sha(BASE/n) for n in ['figure_F08_revised.png','figure_F08_revised.pdf','metrics.json']},
        per_case_mask_scores_recomputed=False,reported_cohort_size=20,instance_selection_cohort_size=10,
        data_origin='The current user message, all 20 numeric pairs transcribed exactly; original mask-based and AI-draft scores preserved separately.')
    save(OUT/'sources.json',sources)
    save(OUT/'validation.json',dict(rows=20,values_in_unit_interval=True,means_match_user_table=True,
        mean_text_only=str(means[0]),mean_photo_assisted=str(means[1]),
        case06_photo_assisted=rows[5][2],case15_lower_photo_score_preserved=rows[14],
        cases_with_higher_photo_iou=sum(a<b for a,b in zip(*values))))
    print('Transcribed 20 pairs; means',*[str(x) for x in means],flush=True)

if __name__=='__main__':main()
