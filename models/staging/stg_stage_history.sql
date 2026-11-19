-- materialized: view
select
    account_id,
    stage,
    case stage
        when 'lead' then 0
        when 'mql' then 1
        when 'sql' then 2
        when 'opportunity' then 3
        when 'won' then 4
    end as stage_order,
    cast(entered_at as timestamp) as entered_at
from {{ source('raw', 'stage_history') }}
