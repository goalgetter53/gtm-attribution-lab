-- materialized: view
select
    opportunity_id,
    account_id,
    primary_contact_id,
    cast(created_at as timestamp) as created_at,
    cast(closed_at as timestamp) as closed_at,
    stage_name,
    cast(is_won as boolean) as is_won,
    cast(amount as double) as amount
from {{ source('raw', 'opportunities') }}
