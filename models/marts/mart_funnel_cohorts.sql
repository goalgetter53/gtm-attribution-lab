-- Stage conversion by monthly lead cohort. Late cohorts are immature (right-censored).
select
    cohort_month,
    count(*) as leads,
    count(mql_at) as mqls,
    count(sql_at) as sqls,
    count(opportunity_at) as opportunities,
    count(won_at) as wins,
    count(mql_at) / nullif(count(*), 0) as lead_to_mql,
    count(sql_at) / nullif(count(mql_at), 0) as mql_to_sql,
    count(opportunity_at) / nullif(count(sql_at), 0) as sql_to_opp,
    count(won_at) / nullif(count(opportunity_at), 0) as opp_to_won,
    count(won_at) / nullif(count(*), 0) as lead_to_won
from {{ ref('fct_funnel_stages') }}
group by cohort_month
order by cohort_month
