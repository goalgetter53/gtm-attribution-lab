-- materialized: view
select
    account_id,
    lower(trim(domain)) as domain,
    industry,
    employee_band,
    cast(created_at as timestamp) as created_at
from {{ source('raw', 'accounts') }}
