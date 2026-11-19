select
    opportunity_id,
    account_id,
    primary_contact_id,
    created_at,
    closed_at,
    stage_name,
    is_won,
    amount,
    closed_at is not null as is_closed,
    case
        when closed_at is not null then (epoch(closed_at) - epoch(created_at)) / 86400.0
    end as cycle_days,
    date_trunc('quarter', created_at) as created_quarter
from {{ ref('stg_opportunities') }}
