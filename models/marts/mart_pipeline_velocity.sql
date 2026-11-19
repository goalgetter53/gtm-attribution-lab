-- Pipeline velocity = (# opportunities x win rate x average ACV) / average sales cycle.
--   # opportunities : opportunities created in the period
--   win rate        : won / closed (open deals excluded from the denominator)
--   average ACV     : mean amount of won deals
--   sales cycle     : mean days from opportunity creation to closed-won
-- Result is expected revenue per day flowing through the pipeline.
with by_quarter as (
    select
        cast(created_quarter as date) as period,
        count(*) as n_opportunities,
        count(*) filter (where is_closed) as n_closed,
        count(*) filter (where is_won) as n_won,
        avg(amount) filter (where is_won) as avg_acv,
        avg(cycle_days) filter (where is_won) as avg_cycle_days
    from {{ ref('fct_opportunities') }}
    group by created_quarter
),

overall as (
    select
        cast(null as date) as period,
        count(*) as n_opportunities,
        count(*) filter (where is_closed) as n_closed,
        count(*) filter (where is_won) as n_won,
        avg(amount) filter (where is_won) as avg_acv,
        avg(cycle_days) filter (where is_won) as avg_cycle_days
    from {{ ref('fct_opportunities') }}
),

unioned as (
    select 'quarter' as grain, * from by_quarter
    union all
    select 'all_time' as grain, * from overall
)

select
    *,
    n_won / nullif(n_closed, 0) as win_rate,
    n_opportunities * (n_won / nullif(n_closed, 0)) * avg_acv / nullif(avg_cycle_days, 0)
        as velocity_per_day
from unioned
order by grain, period
