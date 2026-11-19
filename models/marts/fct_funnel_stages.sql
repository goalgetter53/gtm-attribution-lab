-- One row per account with the CRM timestamp of every funnel stage it reached.
with pivoted as (
    select
        a.account_id,
        a.industry,
        a.employee_band,
        min(case when h.stage = 'lead' then h.entered_at end) as lead_at,
        min(case when h.stage = 'mql' then h.entered_at end) as mql_at,
        min(case when h.stage = 'sql' then h.entered_at end) as sql_at,
        min(case when h.stage = 'opportunity' then h.entered_at end) as opportunity_at,
        min(case when h.stage = 'won' then h.entered_at end) as won_at
    from {{ ref('stg_accounts') }} as a
    left join {{ ref('stg_stage_history') }} as h on h.account_id = a.account_id
    group by a.account_id, a.industry, a.employee_band
)

select
    *,
    date_trunc('month', lead_at) as cohort_month,
    case
        when won_at is not null then 'won'
        when opportunity_at is not null then 'opportunity'
        when sql_at is not null then 'sql'
        when mql_at is not null then 'mql'
        else 'lead'
    end as current_stage
from pivoted
